// Поднять состояние из localStorage/sessionStorage ДО монтирования страниц:
// иначе первая загрузка карточки уходит без цели пользователя.
export default defineNuxtPlugin(() => {
  useSommelier().init();
  useScanner().init();
});
