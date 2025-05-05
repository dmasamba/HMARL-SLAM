import numpy as np
from PIL import Image, ImageDraw
import random
from collections import deque
import os

def generate_cellular_map(width=64, height=48, seed=None, complexity=0.5, save_path="map_ca.png"):
    assert 0.0 <= complexity <= 1.0, "Complexity must be between 0 and 1"

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Map complexity to parameters
    wall_prob = 0.3 + 0.2 * complexity        # 0.3 to 0.5
    steps = int(2 + 4 * complexity)            # 2 to 6
    threshold_floor = int(5 - 2 * complexity)  # 5 to 3
    threshold_wall = int(4 + 2 * complexity)   # 4 to 6

    print(f"Generating map with complexity {complexity:.2f} -> wall_prob={wall_prob:.2f}, steps={steps}, floor_thr={threshold_floor}, wall_thr={threshold_wall}")

    # Initialize map
    map_array = np.random.choice([0, 255], size=(height, width), p=[wall_prob, 1 - wall_prob]).astype(np.uint8)

    def count_wall_neighbors(y, x):
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if map_array[ny][nx] == 0:
                        count += 1
                else:
                    count += 1  # out-of-bounds = wall
        return count

    # Cellular automata smoothing
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

    # Largest connected region (optional cleanup)
    def largest_connected_component(arr):
        visited = np.zeros_like(arr, dtype=bool)
        max_region = []
        for y in range(height):
            for x in range(width):
                if arr[y, x] == 255 and not visited[y, x]:
                    q = deque()
                    region = []
                    q.append((y, x))
                    visited[y, x] = True
                    while q:
                        cy, cx = q.popleft()
                        region.append((cy, cx))
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = cy+dy, cx+dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if arr[ny, nx] == 255 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    if len(region) > len(max_region):
                        max_region = region
        return max_region

    region = largest_connected_component(map_array)
    final_map = np.zeros_like(map_array)
    for y, x in region:
        final_map[y, x] = 255

    # Draw image and marker
    img = Image.fromarray(np.stack([final_map]*3, axis=2))
    draw = ImageDraw.Draw(img)
    if region:
        sy, sx = random.choice(region)
        draw.point((sx, sy), fill=(255, 255, 0))

    img.save(save_path)
    print(f"Map saved to {save_path}")

# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Generate comparison images for increasing sizes (scale up more aggressively)
    base_width, base_height = 32, 24
    num_maps = 8
    scale_step = 32  # Increase width and height by 32 each iteration for larger scaling

    complexities = [0.2, 0.5, 0.8]
    names = ["simple", "medium", "complex"]

    for i in range(num_maps):
        width = base_width + i * scale_step
        height = base_height + i * scale_step
        imgs = []
        tmp_paths = []
        for j, c in enumerate(complexities):
            tmp_path = f"gen_maps/_tmp_map_{i}_{j}.png"
            generate_cellular_map(width=width, height=height, complexity=c, save_path=tmp_path)
            imgs.append(Image.open(tmp_path))
            tmp_paths.append(tmp_path)
        total_width = sum(img.width for img in imgs)
        max_height = max(img.height for img in imgs)
        combined = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in imgs:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        comparison_path = f"gen_maps/comparison_{width}x{height}.png"
        combined.save(comparison_path)
        print(f"Saved side-by-side comparison image to {comparison_path}")
        # Remove temporary files
        for tmp_path in tmp_paths:
            try:
                os.remove(tmp_path)
            except Exception:
                pass