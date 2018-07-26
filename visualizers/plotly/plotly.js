(function () {
  var getColumn = function (array, column, defaultValue){
    var newArr = [];
    for (var j = 0; j<array.length; j++){
      var x = array[j];
      newArr.push (x[column] !== " " ? x[column] : defaultValue);
    };
    return newArr;
  };
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
        posX = header.indexOf('position.x')
        posY = header.indexOf('position.y')
        posZ = header.indexOf('position.z')
        colR = header.indexOf('color.r')
        colG = header.indexOf('color.g')
        colB = header.indexOf('color.b')
        colA = header.indexOf('color.a')
        s = header.indexOf('size')
        labels = header.indexOf('labels')
        console.log(getColumn(dataSeries, posY, 0))
        color = dataSeries.map(function(p){
            return p[colR] !== " " && p[colG] !== " " && p[colB] !== " " && p[colA] !== " " ? 'rgb(' + p[colR] + ',' + p[colG] + ',' + p[colB] + ',' + p[colA] + ')'  : 'rgb(' + 253 + ',' + 106 + ',' + 2 + ',' + 1 + ')' 
        });
        datax = getColumn(dataSeries, posX, 0);
        datay = getColumn(dataSeries, posY, 0);
        dataz = getColumn(dataSeries, posZ, 0);
        sizes = getColumn(dataSeries, s);
        dataLabels = getColumn(dataSeries, labels);
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
