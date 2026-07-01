import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "pyPTV.Dramabox",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "Dramabox_pyPTV") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            // Большое окно для лога генерации — внутри самой ноды.
            const logWidget = node.addWidget("text", "log", "", () => {});
            logWidget.serialize = false;
            logWidget.computeSize = () => [0, 420];
            node._logWidget = logWidget;

            // Подгоняем размер ноды под контент
            requestAnimationFrame(() => {
                const sz = node.computeSize();
                node.setSize([Math.max(node.size[0], 420), sz[1]]);
            });
        };

        const onExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            let text = null;
            if (message?.text) {
                text = Array.isArray(message.text) ? message.text[0] : message.text;
            }
            if (text !== null && this._logWidget) {
                // summary (если есть) кладём в шапку лога
                if (message?.summary) {
                    const sum = Array.isArray(message.summary)
                        ? message.summary[0] : message.summary;
                    if (sum) text = sum + "\n\n" + text;
                }
                this._logWidget.value = text;
                this.setSize([this.size[0], this.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            }
        };
    },
});
