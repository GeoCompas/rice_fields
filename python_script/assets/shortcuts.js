
document.addEventListener('keydown', function (event) {
  if (event.key === 'a') {
    console.log("create windows")
    var button_mark = document.getElementById('mark-window-btn');
    button_mark.click();
  }
  if (event.key === 'd') {
    console.log("Save and Next")
    var button_next = document.getElementById('confirm-btn');
    button_next.click();
  }
});