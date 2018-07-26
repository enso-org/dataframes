(function () {
  var chart = null;
  window.addEventListener("load", function () {
    chart = Plotly.plot("plotly_div");
    // Here things on load
  });
  window.addEventListener("resize", function () {
    // Here things on size change
    Plotly.Plots.resize("plotly_div");
  });


  window.addEventListener("message", function (evt) {
    if(evt.data.data) {
        data = JSON.parse(evt.data.data);
        var dataSeries = data.data;
        header = data.header.map(function(p) {return p.trim();});
        x = header.indexOf('position.x')
        y = header.indexOf('position.y')
        z = header.indexOf('position.z')
        r = header.indexOf('color.r')
        g = header.indexOf('color.g')
        b = header.indexOf('color.b')
        a = header.indexOf('color.a')
        s = header.indexOf('size')
        labels = header.indexOf('labels')

        color = dataSeries.map(function(p){
            return p[r] !== " " && p[g] !== " " && p[b] !== " " && p[a] !== " " ? 'rgb(' + p[r] + ',' + p[g] + ',' + p[b] + ',' + p[a] + ')'  : 'rgb(' + 253 + ',' + 106 + ',' + 2 + ',' + 1 + ')' 
        });
        datax = dataSeries.map(function(p) { return p[x] !== " " ? p[x] : 0; });
        datay = dataSeries.map(function(p) { return p[y] !== " " ? p[y] : 0; });
        dataz = dataSeries.map(function(p) { return p[z] !== " " ? p[z] : 0; });
        sizes = dataSeries.map(function(p) { return p[s]; });
        dataLabels = dataSeries.map(function(p) { return p[labels]; });
        var trace = {
          x: datax,
          y: datay,
          z: dataz,
          text: dataLabels,
          marker: {
            color: color,
            size: sizes
          },
          type: "scatter",
          mode: "markers",
          name: "scatter1"
        };
        // Here things to do with data
        Plotly.newPlot("plotly_div", [trace]);
    };
    });

}());
