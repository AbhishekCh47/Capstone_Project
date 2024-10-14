
$(document).ready(function () {
  "use strict";

  setTimeout(function () {
    $(".loader_bg").fadeToggle();
  }, 0);
  //alert("Tried to make it simple and clean.🏃\nAll kinds of suggestions are acceptable😌");
  //alert("Thanks For Watching :)");

  $(window).on("scroll", function () {
    if ($(document).scrollTop() > 200) {
      $(".navbar").css({
        "background-color": "#333",
        transition: "all 0.8s ease-in-out",
        "box-shadow": "0px 3px 4.6px 0.3px rgba(0, 0, 0, 0.25)",
      });
      $(".navbar-collapse").css("background-color", "transparent");
    } else {
      $(".navbar").css({
        "background-color": "transparent",
        "box-shadow": "none",
      });
      $(".navbar-collapse").css("background-color", "rgba(0,0,0, 0.8)");
    }
  });

  $("#circles").polygonizr();

  startChangingText([".Com", ".Net", ".Org"], 3);

  $("#collapsibleNavbar").click(function () {
    $("#collapsibleNavbar").hide();
  });
  $("#colnav").click(function () {
    $("#collapsibleNavbar").toggle();
  });
});
