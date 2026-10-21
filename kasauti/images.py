import io

from PIL import Image as PILImage
from PIL import ImageOps

from .llm import Image

MAX_SIDE = 2000  # plenty for small label print; keeps vision tokens and upload size down


def prepare_image(data: bytes) -> Image:
    """Fix phone-camera rotation, downscale, and re-encode as JPEG."""
    try:
        img = PILImage.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"Not a readable image: {e}") from e
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return Image(data=out.getvalue(), mime_type="image/jpeg")
