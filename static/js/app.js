// JavaScript to handle mouseover and mouseout events
var activeMethodPill = null;
var activeScenePill = null;
var activeModePill = null;
var activeVidID = 0;
var select = false;


var editor = null;

$(document).ready(function () {
    editor = CodeMirror.fromTextArea(document.getElementById("bibtex"), {
        lineNumbers: false,
        lineWrapping: true,
        readOnly: true
    });
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });

    editor.removeTag = CodeMirror.removeTag;
    var cm = $(".CodeMirror");
    cm.editor = editor;
    editor.save();
    editor.setOption("mode", "htmlmixed");


    // resizeAndPlay($('#sparsity')[0]);
});

function copyBibtex() {
    if (editor) {
        navigator.clipboard.writeText(editor.getValue());
    }
};

// function selectCompVideo(methodPill, scenePill, modePill) {
//     // Your existing logic for video selection
//     // var video = document.getElementById("compVideo");
//     select = true;

//     if (activeMethodPill) {
//         activeMethodPill.classList.remove("active");
//     }
//     if (activeScenePill) {
//         activeScenePill.classList.remove("active");
//     }
//     if (modePill) {
//         activeModePill.classList.remove("active");
//         modePill.classList.add("active");
//         activeModePill = modePill;
//     }
//     activeMethodPill = methodPill;
//     activeScenePill = scenePill;
//     methodPill.classList.add("active");
//     scenePill.classList.add("active");
//     method = methodPill.getAttribute("data-value");
//     pill = scenePill.getAttribute("data-value");
//     mode = activeModePill.getAttribute("data-value");

//     // swap video to avoid flickering
//     activeVidID = 1 - activeVidID;
//     var video_active = document.getElementById("compVideo" + activeVidID);
//     video_active.src = "videos/comparison/" + pill + "_" + method + "_vs_ours_" + mode + ".mp4";
//     video_active.load();
// }