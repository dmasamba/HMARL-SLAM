import numpy as np
from PIL import Image, ImageDraw
import random
import os
from collections import deque


def save_with_marker(map_array, marker_pos, save_path):
    img = Image.fromarray(np.stack([map_array] * 3, axis=2))
    draw = ImageDraw.Draw(img)
    if marker_pos is not None and None not in marker_pos:
        x, y = marker_pos
        draw.point((x, y), fill=(255, 255, 0))
    img.save(save_path)

def generate_cellular_map(width, height, complexity=0.5, seed=None, min_region_size=100, connect_blobs=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Revised parameters
    wall_prob = 0.45 + 0.15 * complexity        # More walls overall
    steps = int(4 + 4 * complexity)             # More smoothing
    threshold_floor = int(5 - 2 * complexity)   # Easy to become floor
    threshold_wall = int(4 + 2 * complexity)    # Hard to stay floor

    # Initialize random noise
    map_array = np.random.choice([0, 255], size=(height, width), p=[wall_prob, 1 - wall_prob]).astype(np.uint8)

    def count_wall_neighbors(y, x):
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if map_array[ny, nx] == 0:
                        count += 1
                else:
                    count += 1
        return count

    # Smoothing steps
    for _ in range(steps):
        new_map = np.copy(map_array)
        for y in range(height):
            for x in range(width):
                walls = count_wall_neighbors(y, x)
                if map_array[y, x] == 255:
                    new_map[y, x] = 0 if walls > threshold_wall else 255
                else:
                    new_map[y, x] = 255 if walls < threshold_floor else 0
        map_array = new_map

    # Flood-fill to get all large regions
    def get_large_components(arr, min_size=50):
        visited = np.zeros_like(arr, dtype=bool)
        regions = []

        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if arr[y, x] == 255 and not visited[y, x]:
                    q = deque()
                    region = []
                    q.append((y, x))
                    visited[y, x] = True
                    while q:
                        cy, cx = q.popleft()
                        region.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < arr.shape[0] and 0 <= nx < arr.shape[1]:
                                if arr[ny, nx] == 255 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    if len(region) >= min_size:
                        regions.append(region)

        final_map = np.zeros_like(arr)
        for region in regions:
            for y, x in region:
                final_map[y, x] = 255
        return final_map, regions

    map_array, regions = get_large_components(map_array, min_size=min_region_size)

    # Optionally connect blobs with thin corridors
    if connect_blobs and len(regions) > 1:
        centroids = []
        for region in regions:
            ys, xs = zip(*region)
            centroids.append((int(np.mean(ys)), int(np.mean(xs))))

        # Connect each blob to the next one
        for (y1, x1), (y2, x2) in zip(centroids, centroids[1:]):
            if random.random() < 0.5:
                map_array[min(y1, y2):max(y1, y2)+1, x1] = 255
                map_array[y2, min(x1, x2):max(x1, x2)+1] = 255
            else:
                map_array[y1, min(x1, x2):max(x1, x2)+1] = 255
                map_array[min(y1, y2):max(y1, y2)+1, x2] = 255

    # Choose a spawn point from any region
    if regions:
        sy, sx = random.choice(random.choice(regions))
    else:
        sy, sx = None, None

    return map_array, (sx, sy)


def generate_room_map(width, height, complexity=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    attempts = int(5 + complexity * 35)
    min_room_ratio = 0.10 - 0.04 * complexity
    max_room_ratio = 0.25 - 0.10 * complexity
    room_min = max(3, int(min(width, height) * min_room_ratio))
    room_max = max(room_min + 1, int(min(width, height) * max_room_ratio))
    jagged_chance = 0.2 + 0.6 * complexity
    corridor_width = max(2, int(min(width, height) * (0.015 + 0.025 * complexity)))

    dungeon = np.zeros((height, width), dtype=np.uint8)
    rooms = []

    for _ in range(attempts):
        w = random.randint(room_min, room_max)
        h = random.randint(room_min, room_max)
        x = random.randint(1, width - w - 1)
        y = random.randint(1, height - h - 1)
        new_room = (x, y, x + w, y + h)

        if all(x + w <= r[0] or x >= r[2] or y + h <= r[1] or y >= r[3] for r in rooms):
            rooms.append(new_room)
            dungeon[y:y + h, x:x + w] = 255

            if len(rooms) > 1:
                prev_x = (rooms[-2][0] + rooms[-2][2]) // 2
                prev_y = (rooms[-2][1] + rooms[-2][3]) // 2
                new_x = (x + x + w) // 2
                new_y = (y + y + h) // 2

                if random.random() < jagged_chance:
                    y1, y2 = sorted([prev_y, new_y])
                    dungeon[y1:y2 + 1, prev_x - corridor_width // 2:prev_x + corridor_width // 2 + 1] = 255
                    x1, x2 = sorted([prev_x, new_x])
                    dungeon[new_y - corridor_width // 2:new_y + corridor_width // 2 + 1, x1:x2 + 1] = 255
                else:
                    x1, x2 = sorted([prev_x, new_x])
                    dungeon[prev_y - corridor_width // 2:prev_y + corridor_width // 2 + 1, x1:x2 + 1] = 255
                    y1, y2 = sorted([prev_y, new_y])
                    dungeon[y1:y2 + 1, new_x - corridor_width // 2:new_x + corridor_width // 2 + 1] = 255

    if rooms:
        x = (rooms[0][0] + rooms[0][2]) // 2
        y = (rooms[0][1] + rooms[0][3]) // 2
    else:
        x, y = None, None

    return dungeon, (x, y)


def generate_and_save(style='room', width=64, height=48, complexity=0.5, seed=None, save_path="output.png"):
    if style == 'cellular':
        map_array, start = generate_cellular_map(width, height, complexity, seed)
    elif style == 'room':
        map_array, start = generate_room_map(width, height, complexity, seed)
    else:
        raise ValueError(f"Unknown style '{style}'")
    save_with_marker(map_array, start, save_path)


# Example batch usage
if __name__ == "__main__":
    os.makedirs("gen_maps", exist_ok=True)

    base_width, base_height = 32, 24
    num_maps = 6
    scale_step = 32
    complexities = [0.2, 0.5, 0.8]
    styles = ['room', 'cellular']

    for style in styles:
        for i in range(num_maps):
            width = base_width + i * scale_step
            height = base_height + i * scale_step
            imgs = []
            tmp_paths = []
            for j, c in enumerate(complexities):
                tmp_path = f"gen_maps/_tmp_{style}_{i}_{j}.png"
                generate_and_save(style=style, width=width, height=height, complexity=c, save_path=tmp_path)
                imgs.append(Image.open(tmp_path))
                tmp_paths.append(tmp_path)
            total_width = sum(img.width for img in imgs)
            max_height = max(img.height for img in imgs)
            combined = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in imgs:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            comparison_path = f"gen_maps/{style}_comparison_{width}x{height}.png"
            combined.save(comparison_path)
            print(f"Saved {style} comparison to {comparison_path}")
            for tmp in tmp_paths:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
