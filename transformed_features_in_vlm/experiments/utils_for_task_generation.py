from PIL import Image, ImageDraw
import random
import math

def generate_random_shape_image(shape_type, color, background_color="white", output_path=None):
    width, height = 500, 500
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    center_x, center_y = width // 2, height // 2
    white_line_width = 10
    if shape_type == "polygon":
        num_sides = random.randint(3, 8)
        min_radius, max_radius = 50, 200
        radius = random.randint(min_radius, max_radius)

        points = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        draw.polygon(points, fill=color)
        draw.polygon(points, outline="white", width=white_line_width)

    elif shape_type == "circle":
        radius = random.randint(50, 200)
        bounding_box = [
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius
        ]

        draw.ellipse(bounding_box, fill=color)
        draw.ellipse(bounding_box, outline="white", width=white_line_width)
    
    elif shape_type == "ellipse":
        radius_x = random.randint(50, 200)
        radius_y = random.randint(50, 200)
        bounding_box = [
            center_x - radius_x, center_y - radius_y,
            center_x + radius_x, center_y + radius_y
        ]
        draw.ellipse(bounding_box, fill=color)
        draw.ellipse(bounding_box, outline="white", width=white_line_width)
    
    else:
        raise ValueError("Acceptable Shape Types: 'polygon', 'circle', 'ellipse'")
    
    if output_path is not None:
        image.save(output_path)
    return image