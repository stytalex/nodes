import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "pyPTV.ImageBatchUpload",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "LTX23ImageBatchUpload") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const node = this;

            // ----------------------------------------------------------------
            // Кнопка Upload
            // ----------------------------------------------------------------
            const uploadBtn = node.addWidget("button", "📁 Upload Images", null, () => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/jpeg,image/png,image/webp,image/bmp";
                input.multiple = true;

                input.onchange = async () => {
                    const files = Array.from(input.files);
                    if (!files.length) return;

                    // Получаем output_folder из виджета
                    const outputFolderWidget = node.widgets.find(w => w.name === "output_folder");
                    const outputFolder = outputFolderWidget ? outputFolderWidget.value : "/tmp/dataset";

                    await uploadFiles(node, files, outputFolder);
                };

                input.click();
            });

            uploadBtn.serialize = false;

            // ----------------------------------------------------------------
            // Прогресс-бар (скрытый по умолчанию)
            // ----------------------------------------------------------------
            const progressWidget = node.addWidget("text", "status", "", () => {});
            progressWidget.inputEl = null;
            progressWidget.serialize = false;
            progressWidget.computeSize = () => [node.size[0], 22];

            // Скрываем пока нет загрузки
            progressWidget.value = "";

            // ----------------------------------------------------------------
            // Скрытый виджет uploaded_files — передаёт список файлов в Python
            // ----------------------------------------------------------------
            let uploadedFilesWidget = node.widgets.find(w => w.name === "uploaded_files");
            if (!uploadedFilesWidget) {
                uploadedFilesWidget = node.addWidget("text", "uploaded_files", "[]", () => {});
                uploadedFilesWidget.serialize = true;
            }
            // Прячем его от пользователя
            uploadedFilesWidget.type = "hidden";
            uploadedFilesWidget.computeSize = () => [0, 0];

            node._progressWidget = progressWidget;
            node._uploadedFilesWidget = uploadedFilesWidget;
        };
    },
});


// ----------------------------------------------------------------------------
// Загрузка файлов
// ----------------------------------------------------------------------------

async function uploadFiles(node, files, outputFolder) {
    const progressWidget = node._progressWidget;
    const uploadedFilesWidget = node._uploadedFilesWidget;

    const total = files.length;
    const uploadedNames = [];

    setStatus(progressWidget, `Загрузка 0 / ${total}...`, "#aaaaff");

    for (let i = 0; i < files.length; i++) {
        const file = files[i];

        try {
            const formData = new FormData();
            formData.append("image", file);
            formData.append("overwrite", "true");

            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
            });

            if (!resp.ok) {
                console.error(`[pyPTV] Ошибка загрузки ${file.name}:`, resp.statusText);
                setStatus(progressWidget, `Ошибка: ${file.name}`, "#ff6666");
                continue;
            }

            const data = await resp.json();
            // ComfyUI возвращает { name: "filename.jpg", subfolder: "", type: "input" }
            uploadedNames.push(data.name);

            const pct = Math.round(((i + 1) / total) * 100);
            setStatus(progressWidget, `Загрузка ${i + 1} / ${total} (${pct}%)`, "#aaffaa");

        } catch (err) {
            console.error(`[pyPTV] Исключение при загрузке ${file.name}:`, err);
        }
    }

    if (!uploadedNames.length) {
        setStatus(progressWidget, "Не удалось загрузить файлы", "#ff6666");
        return;
    }

    // ----------------------------------------------------------------
    // Финализация — копируем в папку датасета и нумеруем
    // ----------------------------------------------------------------
    setStatus(progressWidget, "Сохранение в датасет...", "#ffddaa");

    try {
        const resp = await api.fetchApi("/pyptv/image_batch_finalize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filenames: uploadedNames,
                output_folder: outputFolder,
            }),
        });

        const result = await resp.json();

        if (result.errors && result.errors.length) {
            console.warn("[pyPTV] Ошибки при финализации:", result.errors);
        }

        // Передаём список файлов в Python ноду через hidden виджет
        uploadedFilesWidget.value = JSON.stringify(uploadedNames);

        setStatus(
            progressWidget,
            `✅ Готово: ${result.copied} / ${result.total} файлов`,
            "#aaffaa"
        );

        // Подстраиваем размер ноды
        node.setSize([node.size[0], node.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);

    } catch (err) {
        console.error("[pyPTV] Ошибка финализации:", err);
        setStatus(progressWidget, "Ошибка финализации", "#ff6666");
    }
}


// ----------------------------------------------------------------------------
// Вспомогательная функция статуса
// ----------------------------------------------------------------------------

function setStatus(widget, text, color) {
    widget.value = text;
    widget._color = color || "#ffffff";
    // Перерисовываем канвас
    app.graph.setDirtyCanvas(true, false);
}
