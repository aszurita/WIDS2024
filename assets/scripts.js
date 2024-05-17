let slideIndex = 0;
let slides = document.getElementsByClassName("slide");
let dots = document.getElementsByClassName("dot");

function showSlides() {
    for (let i = 0; i < slides.length; i++) {
        slides[i].style.left = "100%"; // Hide all slides
    }
    slideIndex++;
    if (slideIndex > slides.length) { slideIndex = 1 }
    slides[slideIndex - 1].style.left = "0"; // Show the current slide
    for (let i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }
    dots[slideIndex - 1].className += " active";
    setTimeout(showSlides, 8000); // Change image every 8 seconds
}

function currentSlide(n) {
    slideIndex = n - 1;
    showSlides();
}

document.addEventListener("DOMContentLoaded", function () {
    showSlides(); // Start the slideshow when the document is loaded
});
