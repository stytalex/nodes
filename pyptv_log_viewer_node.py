"""
Log Viewer (pyPTV)
Универсальная нода для отображения текста / лога от любой другой ноды.
Большое многострочное окно.
"""


class PyPTVLogViewer:
    """Отображает входной текст в большом окне."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    CATEGORY = "pyPTV"
    OUTPUT_NODE = True

    def show(self, text: str):
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "PyPTVLogViewer": PyPTVLogViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyPTVLogViewer": "Log Viewer (pyPTV)",
}
