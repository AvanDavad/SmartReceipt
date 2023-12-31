from PIL import ImageFont

def get_font(font_size):
    font = ImageFont.truetype(
        "LiberationMono-Regular.ttf",
        font_size,
    )
    return font
