"""
Log Viewer (pyPTV)
═══════════════════════════════════════════════════════════════════════════════
Универсальная нода для отображения текста / лога от любой другой ноды.

Как работает:
  • Подключи STRING-выход любой ноды ко входу text.
  • После выполнения текст отобразится в большом окне ноды (600px).
  • Текст остаётся висеть до следующего запуска — удобно следить за статусом.

Примеры использования:
  • Load Dataset → log → Log Viewer  (покажет список скачанных файлов)
  • Любая нода со STRING-выходом → Log Viewer

Вход:
  • text — любой текст / лог (STRING)

Выход:
  • text — тот же текст, можно цеплять дальше
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
