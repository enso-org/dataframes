module.exports = function (t) {
    var placeholderVis = {name: "image", path: "image.html"};
    var types = ["Plot", "Chart", "Histogram"];
    return types.includes(t.constructor) ? [placeholderVis] : [];
};
