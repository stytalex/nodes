import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "pyPTV.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "LTX23LoadDataset") return;

        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig?.apply(this, arguments);
            // Убираем лишнюю высоту — оставляем только под контент
            requestAnimationFrame(() => {
                const sz = this.computeSize();
                this.setSize([this.size[0], sz[1]]);
            });
        };
    },
});
