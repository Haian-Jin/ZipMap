// Written by Dor Verbin, December 2023, based on video_comparison.js, which is based on:
// http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// with additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId, container, targetHeight = 2000) {
  var videoMerge = document.getElementById(videoId + "Merge");
  var slider = document.getElementById(videoId + "Slider");
  var vid = document.getElementById(videoId);


  var mergeContext = videoMerge.getContext("2d");
  if (vid.readyState > 3) {
    vid.play();

    function drawLoop() {
      var vidWidthOrig = vid.videoWidth;
      var vidHeightOrig = vid.videoHeight;
      var vidWidth = container.width();
      var vidHeight = vidWidth * vidHeightOrig / vidWidthOrig / 7;

      mergeContext.drawImage(vid, 0, slider.value * vidHeightOrig / 7, vidWidthOrig, vidHeightOrig / 7, 0, 0, vidWidth, vidHeight);
      requestAnimationFrame(drawLoop);


    }
    requestAnimationFrame(drawLoop);
  }
}

Number.prototype.clamp = function (min, max) {
  return Math.min(Math.max(this, min), max);
};


function resizeAndPlay(videoElement, targetHeight = 2000) {
  var canvas = document.getElementById(videoElement.id + "Merge");
  var slider = document.getElementById(videoElement.id + "Slider");
  const sliderValue = slider.value;
  const container = $('.video-compare-container');

  console.log(sliderValue)

  function resizerbin() {
    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight / 7;
    const canvasWidth = container.width();
    const canvasHeight = canvasWidth * videoHeight / videoWidth;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
  }

  resizerbin();

  console.log('play when ready', videoElement)
  if (videoElement.readyState >= 3) {
    videoElement.play();
  } else if (!self.readyStateListenerAttached) {
    document.addEventListener('readystatechange', function () {
      if (videoElement.readyState >= 3) {
        videoElement.play();
      }
    });
  }

  // videoElement.play();
  videoElement.style.height = "0px";  // Hide video without stopping it

  playVids(videoElement.id, container, targetHeight);

  console.log(canvas);

  $(window).on('resize', resizerbin);
}


// From: https://benfrain.com/automatically-play-and-pause-video-as-it-enters-and-leaves-the-viewport-screen/
function playPauseVideo() {
  let videos = document.querySelectorAll("video");
  videos.forEach((video) => {
      // We can only control playback without insteraction if video is mute
      video.muted = true;
      // Play is a promise so we need to check we have it
      let playPromise = video.play();
      if (playPromise !== undefined) {
          playPromise.then((_) => {
              let observer = new IntersectionObserver(
                  (entries) => {
                      entries.forEach((entry) => {
                          if (
                              entry.intersectionRatio !== 1 &&
                              !video.paused
                          ) {
                              video.pause();
                          } else if (video.paused) {
                              video.play();
                          }
                      });
                  },
                  { threshold: 0.5 }
              );
              observer.observe(video);
          });
      }
  });
}