document.getElementById("Enter").addEventListener("click", async function() {
    console.log("clicked")
    var start = document.getElementById("text1").value;
    var end = document.getElementById("text2").value;
    if (start.startsWith("2")){
        document.getElementById("blueprint").src="./Blueprints/2-2.jpg";
        console.log("2");
    } else if (start.startsWith("1")){
        document.getElementById("blueprint").src="./Blueprints/1-1.png";
        console.log("1");
    }
    const query = await fetch(`http://127.0.0.1:5000/${start}/${end}`, { method: 'GET' })
    .then(response => response.blob())
    .then(blob => {
        console.log(blob);
        var exists = document.getElementById("graph");
        if (exists){
            exists.remove();
        }
        var graph = document.createElement("img");
        graph.setAttribute("src", URL.createObjectURL(blob));
        graph.setAttribute("id", "graph");
        document.getElementById("pics").appendChild(graph);
    }).catch(error => {console.log(error)});
    // data = await query.json();
    console.log(query);
});


// document.getElementById("explore").addEventListener("click", function{

// })

