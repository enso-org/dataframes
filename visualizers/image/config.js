module.exports = function (t) {
    var placeholderVis = {name: "plot", path: "image.html"};
    var types = ["Plot", "Chart", "Histogram", "KDE2", "KDE"];
    return types.includes(t.constructor) ? [placeholderVis] : [];
};
