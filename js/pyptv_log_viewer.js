import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "pyPTV.LogViewer",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "PyPTVLogViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            // Скрываем стандартный input-виджет text полностью
            const textWidget = node.widgets.find(w => w.name === "text");
            if (textWidget) {
                textWidget.type = "hidden";
                textWidget.serialize = false;
                textWidget.computeSize = () => [0, -4];
            }

            // Большой виджет для отображения лога
            const logWidget = node.addWidget("text", "log", "", () => {});
            logWidget.serialize = false;
            logWidget.computeSize = () => [0, 500];
            node._logWidget = logWidget;

            // Устанавливаем размер ноды под контент
            const sz = node.computeSize();
            node.setSize([Math.max(node.size[0], 320), sz[1]]);
        };

        const onExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            let text = null;
            if (message?.text) {
                text = Array.isArray(message.text) ? message.text[0] : message.text;
            }
            if (text !== null && this._logWidget) {
                this._logWidget.value = text;
                this.setSize([this.size[0], this.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            }
        };
    },
});
