document.querySelector("form").addEventListener("submit", function() {
  document.querySelector("button").innerHTML = "Processing...";
  document.querySelector("button").disabled = true;
});