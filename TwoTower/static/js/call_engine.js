var selectedEngine = "LightFM"; 

function sendInput() {
    var inputValue = document.getElementById("inputText").value;
    var selectedOption = selectedEngine;
    console.log(selectedOption);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/process_id", true);
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");

    xhr.onload = function() {
        if (xhr.status === 200) {
            var resultBox = document.getElementById("result");
            var response = JSON.parse(xhr.responseText);
            resultBox.innerHTML = "<p style='color: white; font-weight: bold;'>Result: " + response.result + "</p>";

        }
    };

    xhr.send("inputValue=" + inputValue + "&selectedOption=" + selectedOption);

}

function selectOption(value) {
    //document.getElementById("dropdownButton").innerText = value;
    selectedEngine = value;
}