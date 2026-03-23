async function getExplanation() {
    const question = document.getElementById("question").value;
    const level = document.getElementById("level").value;
    const style = document.getElementById("style").value;

    document.getElementById("output").innerText = "Thinking...";

    const res = await fetch("/explain", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ question, level, style })
    });

    const data = await res.json();
    document.getElementById("output").innerText = data.explanation;
}
