// Get modal element
var navLeft = document.getElementById("navLeft"); 
var toggleNavBtn = document.getElementById("toggleNav");


// Advanced  On/Off nav-left
function toggleNav() {
    if (navLeft.style.display === "none" || navLeft.style.display === "") {
        navLeft.style.display = "block";
        toggleNavBtn.textContent = "Off"; 
    } else {
        navLeft.style.display = "none";
        toggleNavBtn.textContent = "On"; 
    }
} 
toggleNavBtn.addEventListener("click", toggleNav);

navLeft.style.display = "block";
toggleNavBtn.textContent = "Off";

//   Filter slider
function toggleSlider() {
    const sliderContainer = document.querySelector('.slider-container');
    sliderContainer.style.display = (sliderContainer.style.display === 'none' || sliderContainer.style.display === '') ? 'flex' : 'none';
}

function updateValue() {
    const slider = document.getElementById('slider');
    const ImageBlocks = document.querySelectorAll('.image-block');
    const sliderValue = document.getElementById('slider-value');
    sliderValue.textContent = slider.value;
    ImageBlocks.forEach(block => {
        block.style.width = `${slider.value}px`; 
    });
}  

// Object and color search
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('.search-icon');
    const iconContainer = document.querySelector('.icon-container');

    searchInput.addEventListener('input', function() {
        const searchTerm = searchInput.value.toLowerCase();
        
        const iconBoxes = iconContainer.querySelectorAll('.icon-box');
        
        iconBoxes.forEach(iconBox => {
            const nameIcon = iconBox.querySelector('.name-icon').textContent.toLowerCase();
            if (nameIcon.includes(searchTerm)) {
                iconBox.style.display = '';
            } else {
                iconBox.style.display = 'none';
            }
        });
    });

    const searchColor = document.querySelector('.search-color');
    const colorContainer = document.querySelector('.color-container');

    searchColor.addEventListener('input', function() {
        const searchTerm = searchColor.value.toLowerCase();
        
        const colorBoxes = colorContainer.querySelectorAll('.color-box');
        
        colorBoxes.forEach(colorBox => {
            const nameColor = colorBox.querySelector('.name-color').textContent.toLowerCase();
            if (nameColor.includes(searchTerm)) {
                colorBox.style.display = '';
            } else {
                colorBox.style.display = 'none';
            }
        });
    });
});


// change tab
const videoContent = document.getElementById('videoContent');
const frameContent = document.getElementById('frameContent');
const videoBtn = document.getElementById('showVideo');
const frameBtn = document.getElementById('showFrame');

videoBtn.addEventListener('click', () => {
    videoContent.classList.add('active');
    frameContent.classList.remove('active');
    videoBtn.classList.add('active-change');
    frameBtn.classList.remove('active-change');
});

frameBtn.addEventListener('click', () => {
    frameContent.classList.add('active');
    videoContent.classList.remove('active');
    frameBtn.classList.add('active-change');
    videoBtn.classList.remove('active-change');
});

// tag search bboxes
function dragInit(elemToDrag) {
    let pos3 = 0, pos4 = 0;

    elemToDrag.onmousedown = function(e) {
        if (e.target.tagName === 'INPUT' || e.target.classList.contains('tag-search-bboxes-close')) {
            return;
        }

        e.preventDefault();
        pos3 = e.clientX;
        pos4 = e.clientY;

        document.onmousemove = function(e) {
            e = e || window.event;
            e.preventDefault();

            let pos1 = pos3 - e.clientX;
            let pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;

            elemToDrag.style.top = (elemToDrag.offsetTop - pos2) + "px";
            elemToDrag.style.left = (elemToDrag.offsetLeft - pos1) + "px";
        };

        document.onmouseup = function() {
            document.onmouseup = null;
            document.onmousemove = null;
        };
    };
}

document.addEventListener('DOMContentLoaded', (event) => {
    const elemToDrag = document.querySelector('.tag-search-bboxes-container');
    dragInit(elemToDrag);
});

const addTagBboxes = document.querySelector('.add-tag-bboxes');
const tagSearchBboxesClose = document.querySelector('.tag-search-bboxes-close');
const tagSearchBboxesContainer = document.querySelector('.tag-search-bboxes-container');

addTagBboxes.addEventListener('click', (event) => {
    const computedStyle = window.getComputedStyle(tagSearchBboxesContainer);
    if (computedStyle.display === 'none') {
        tagSearchBboxesContainer.style.display = 'block';
        document.querySelector(".tag-search-bboxes").focus();
    } else {
        tagSearchBboxesContainer.style.display = 'none';
    }
});

tagSearchBboxesClose.addEventListener('click', (event) => {
    tagSearchBboxesContainer.style.display = 'none';
});

// change tab main
const proposeContent = document.getElementById('proposeContent');
const imageContent = document.getElementById('imageContent');
const imageBtn = document.getElementById('showImage');
const proposeBtn = document.getElementById('showPropose');

imageBtn.addEventListener('click', () => { 
    proposeContent.style.display = "none";
    imageContent.style.display = "block";
    imageBtn.classList.add('main-change-active');
    proposeBtn.classList.remove('main-change-active');
});

proposeBtn.addEventListener('click', () => {
    proposeContent.style.display = "block";
    imageContent.style.display = "none";
    proposeBtn.classList.add('main-change-active');
    imageBtn.classList.remove('main-change-active');
});
