import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "pyPTV.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "LTX23LoadDataset") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const node = this;

            // Текстовый виджет для отображения статуса
            const statusWidget = node.addWidget("text", "status", "", () => {});
            statusWidget.serialize = false;
            statusWidget.inputEl = null;
            statusWidget.computeSize = () => [node.size[0], 400];
            node._statusWidget = statusWidget;
        };

        const onExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (message?.status && this._statusWidget) {
                this._statusWidget.value = message.status[0];
                this.setSize([this.size[0], this.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            }
        };
    },
});
