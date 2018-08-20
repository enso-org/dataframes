(function () {
  var entityMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
    '/': '&#x2F;',
    '`': '&#x60;',
    '=': '&#x3D;'
  };

  var escapeHtml = function (string) {
    return string.replace(/[&<>"'`=\/]/g, function (s) {
      return entityMap[s];
    });
  }

  var render = function (data) {
    document.body.innerHTML = '<div class="container small"><img src=' + data.small + " /></div>" +
                              '<div class="container big"><img src=' + data.big + " /></div>";
  }

  window.addEventListener("message", function (evt) {
    d = JSON.parse(evt.data.data);
    render(d);
  });
}());
