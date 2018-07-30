(function () {
  var defPos = 0;
  var defColR = 253;
  var defColG = 106;
  var defColB = 2;
  var defColA = 1;
  var getColumn = function (array, column, defaultValue){
    var newArr = [];
    for (var j = 0; j<array.length; j++){
      var x = array[j];
      newArr.push (x[column] !== " " ? x[column] : defaultValue);
    };
    return newArr;
  };
  var getColor = function (array, r,g,b,a, defR, defG, defB, defA){
    var newColorArr = [];
    for (var j = 0; j<array.length; j++){
      var x = array[j];
      newColorArr.push (x[r] !== " " && x[g] !== " " && x[b] !== " " && x[a] !== " " ? 'rgb(' + x[r] + ',' + x[g] + ',' + x[b] + ',' + x[a] + ')'  : 'rgb(' + defR + ',' + defG + ',' + defB + ',' + defA + ')' );
    }; 
    return newColorArr;
  }

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
        color = getColor(dataSeries,colR, colG, colB, colA, defColR, defColG, defColB, defColA)
        datax = getColumn(dataSeries, posX, defPos);
        datay = getColumn(dataSeries, posY, defPos);
        dataz = getColumn(dataSeries, posZ, defPos);
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
