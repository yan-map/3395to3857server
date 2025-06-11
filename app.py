import os
import aiohttp
import asyncio
from fastapi import FastAPI, Response
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import numpy as np
from io import BytesIO
import pyproj
from functools import lru_cache
import logging
import redis

# Настройка FastAPI
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# URL-маска для тайлов Яндекса
YANDEX_TILE_URL = "https://sat02.maps.yandex.net/tiles?l=sat&v=3.1726.0&x={x}&y={y}&z={z}&lang=ru_KZ&client_id=yandex-web-maps"

# Настройки кэширования
CACHE_DIR = "/app/tile_cache"
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
if USE_REDIS:
    redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Создание папки для кэша
os.makedirs(CACHE_DIR, exist_ok=True)

# Преобразователь координат EPSG:3395 -> EPSG:3857
transformer = pyproj.Transformer.from_crs("EPSG:3395", "EPSG:3857", always_xy=True)

# Функция для вычисления границ тайла в EPSG:3395
def get_tile_bounds(z, x, y, tile_size=256, extent=20037508.342789244):
    res = 2 * extent / (2 ** z) / tile_size
    left = -extent + x * res * tile_size
    right = left + res * tile_size
    top = extent - y * res * tile_size
    bottom = top - res * tile_size
    # Преобразуем границы из EPSG:3857 в EPSG:3395 (обратное преобразование)
    left_3395, bottom_3395 = transformer.transform(left, bottom, direction="INVERSE")
    right_3395, top_3395 = transformer.transform(right, top, direction="INVERSE")
    return left_3395, bottom_3395, right_3395, top_3395

# Функция для загрузки тайла
async def fetch_tile(session, x, y, z):
    url = YANDEX_TILE_URL.format(x=x, y=y, z=z)
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            return None
    except Exception as e:
        logging.error(f"Fetch tile error: {e}")
        return None

# Функция для перепроецирования тайла
def reproject_tile(tile_data, z, x, y, src_crs="EPSG:3395", dst_crs="EPSG:3857"):
    try:
        # Открываем изображение как растр
        with rasterio.MemoryFile(tile_data) as memfile:
            with memfile.open(driver='PNG') as src:
                # Получаем данные изображения
                data = src.read()
                if data.shape[0] == 4:  # Если есть альфа-канал, убираем его
                    data = data[:3]
                
                # Вычисляем границы тайла
                left, bottom, right, top = get_tile_bounds(z, x, y)
                transform = from_bounds(left, bottom, right, top, src.width, src.height)

                # Подготовка параметров для перепроецирования
                dst_transform, width, height = rasterio.warp.calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, left=left, bottom=bottom, right=right, top=top
                )
                kwargs = {
                    'driver': 'PNG',
                    'height': height,
                    'width': width,
                    'count': data.shape[0],
                    'dtype': data.dtype,
                    'crs': dst_crs,
                    'transform': dst_transform
                }

                # Перепроецирование
                with rasterio.MemoryFile() as mem_dst:
                    with mem_dst.open(**kwargs) as dst:
                        for i in range(data.shape[0]):
                            reproject(
                                source=data[i],
                                destination=rasterio.band(dst, i + 1),
                                src_transform=transform,
                                src_crs=src_crs,
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.bilinear
                            )
                        return dst.read(), dst_transform
    except Exception as e:
        logging.error(f"Reprojection error: {e}")
        return None, None

# Функция для сшивания двух тайлов
def stitch_tiles(tile1_data, tile2_data, offset_x=0, offset_y=0):
    try:
        tile1 = Image.open(BytesIO(tile1_data))
        tile2 = Image.open(BytesIO(tile2_data))
        stitched = Image.new("RGB", (256, 256))
        stitched.paste(tile1, (0, 0))
        stitched.paste(tile2, (int(offset_x), int(offset_y)))
        output = BytesIO()
        stitched.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logging.error(f"Stitching error: {e}")
        return None

# Функция для вычисления координат тайлов
@lru_cache(maxsize=1000)
def get_yandex_tiles(z, x, y):
    tile_size = 256
    extent = 20037508.342789244
    res = 2 * extent / (2 ** z) / tile_size
    x_center = -extent + x * res * tile_size + res * tile_size / 2
    y_center = extent - y * res * tile_size - res * tile_size / 2
    x_3395, y_3395 = transformer.transform(x_center, y_center)
    tile_x1 = int((x_3395 + extent) / (2 * extent) * (2 ** z))
    tile_y1 = int((extent - y_3395) / (2 * extent) * (2 ** z))
    tile_x2 = tile_x1 + 1 if x_3395 < 0 else tile_x1 - 1
    tile_y2 = tile_y1 + 1 if y_3395 < 0 else tile_y1 - 1
    return (tile_x1, tile_y1), (tile_x2, tile_y2)

# Функция для кэширования
def cache_tile(z, x, y, data):
    try:
        if USE_REDIS:
            redis_client.setex(f"tile:{z}:{x}:{y}", 3600, data)  # Кэш на 1 час
        else:
            cache_path = os.path.join(CACHE_DIR, f"{z}/{x}/{y}.png")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(data)
    except Exception as e:
        logging.error(f"Cache error: {e}")

defのですdef get_cached_tile(z, x, y):
    try:
        if USE_REDIS:
            return redis_client.get(f"tile:{z}:{x}:{y}")
        cache_path = os.path.join(CACHE_DIR, f"{z}/{x}/{y}.png")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()
        return None
    except Exception as e:
        logging.error(f"Get cache error: {e}")
        return None

# Маршрут для получения тайлов
@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    # Проверка кэша
    cached_tile = get_cached_tile(z, x, y)
    if cached_tile:
        return Response(content=cached_tile, media_type="image/png")

    # Загрузка тайлов
    async with aiohttp.ClientSession() as session:
        (tile_x1, tile_y1), (tile_x2, tile_y2) = get_yandex_tiles(z, x, y)
        tile1_task = fetch_tile(session, tile_x1, tile_y1, z)
        tile2_task = fetch_tile(session, tile_x2, tile_y2, z)
        tile1_data, tile2_data = await asyncio.gather(tile1_task, tile2_task)

        if tile1_data and tile2_data:
            # Перепроецирование
            tile1_reprojected, _ = reproject_tile(tile1_data, z, tile_x1, tile_y1)
            tile2_reprojected, _ = reproject_tile(tile2_data, z, tile_x2, tile_y2)
            if tile1_reprojected is None or tile2_reprojected is None:
                return Response(status_code=500, content="Reprojection failed")

            # Сшивание
            stitched_tile = stitch_tiles(tile1_data, tile2_data)
            if stitched_tile is None:
                return Response(status_code=500, content="Stitching failed")

            # Сохранение в кэш
            cache_tile(z, x, y, stitched_tile)

            return Response(content=stitched_tile, media_type="image/png")
        return Response(status_code=404, content="Failed to fetch tiles")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))