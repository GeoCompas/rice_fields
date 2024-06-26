document.addEventListener('keydown', function (event) {
  const key = event.key;
  let idButton;

  switch (key) {
    case 'a':
      idButton = 'mark-window-btn';
      break;
    case 'd':
      idButton = 'confirm-btn';
      break;
    case 'w':
      idButton = 'rm-last-window-btn';
      break;
    default:
      idButton = null
      break
  }
  if (idButton) {
    var button_mark = document.getElementById(idButton);
    button_mark.click();
  }
});