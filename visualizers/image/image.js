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

  var displayImage = function (data) {
    document.body.innerHTML = "<img src=" + data + " />"
  }

  var render = function (data) {
      displayImage(data);
  };

  window.addEventListener("message", function (evt) {
    render(evt.data.data);
  });
}());
