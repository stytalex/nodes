import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "pyPTV.LogViewer",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "PyPTVLogViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const node = this;

            // Скрываем стандартный виджет text (он маленький)
            const textWidget = node.widgets.find(w => w.name === "text");
            if (textWidget) {
                textWidget.type = "hidden";
                textWidget.computeSize = () => [0, 0];
            }

            // Большой виджет для лога
            const logWidget = node.addWidget("text", "log", "", () => {});
            logWidget.serialize = false;
            logWidget.inputEl = null;
            logWidget.computeSize = () => [node.size[0], 600];
            node._logWidget = logWidget;
        };

        const onExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (message?.text && this._logWidget) {
                this._logWidget.value = message.text[0];
                this.setSize([this.size[0], this.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            }
        };
    },
});
