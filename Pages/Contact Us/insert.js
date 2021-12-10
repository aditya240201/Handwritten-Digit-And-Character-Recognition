

let Email, Msg;

function preventFromSubmitting(event){
    
    event.preventDefault();
}

function readForm(){
    Email = document.getElementById("email").value;
    Msg = document.getElementById("message").value;
    console.log("Email is -->" + Email)
    console.log("message is--> " + Msg)
    

}

document.getElementById("submit").onclick = function(){
    readForm();
    firebase
    .database()
    .ref("student/" + Email)
    .set({
        email : Email,
        message: Msg,

    });
    alert("Data Inserted");
    document.getElementById("email").value = "";
    document.getElementById("message").value = "";

};

