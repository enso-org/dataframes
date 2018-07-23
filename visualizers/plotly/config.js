var cfgHelper = require("../visualization-config-helper.js")

module.exports = function (type) {
    var plotlyPattern =
        { constructor: ["Stream", "List", "Dataframe"], fields: [{constructor: ["Int", "Real"], fields: { any: true }}]
        };

    if (cfgHelper.matchesType(type, plotlyPattern))
        {
            return [{name: "plotly", path: "plotly.html"}]
        }
    else
        return [];
};
