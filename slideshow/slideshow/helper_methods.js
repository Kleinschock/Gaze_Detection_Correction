const uid = function (i) {
    return function () {
        return "generated_id-" + (++i);
    };
}(0);

// Note: We're no longer using rotateRotatables as the rotate gesture
// now handles vertical navigation instead of image rotation

const zoom = function(zoomStepSize) {
  body = document.getElementsByTagName("body")[0];
  const currentZoom = Number(body.style.zoom.replace("%", "")) || 100;
  newZoom = Math.max(currentZoom + zoomStepSize, 40); // don't go lower than 40% zoom
  body.style.zoom = newZoom + "%"
}