let startTime;
let reactionTime;
let isPressVisible = false;
let imageTimerId;
let audioTimerId;
let pressTimerId;
let reactionTimeArrayBase = [];
let reactionTimeArrayDistractors = [];
let flag = false;
let hasPressed = false;
let concludedFlag = false;

const imagePaths = [
    staticUrls.images.car,
    staticUrls.images.dice,
    staticUrls.images.giraffe,
    staticUrls.images.phone,
    staticUrls.images.shovel,
    staticUrls.images.pen
];

const audioPaths = [
    staticUrls.audio.bird,
    staticUrls.audio.dice,
    staticUrls.audio.doorKnock,
    staticUrls.audio.elephant,
    staticUrls.audio.motorola,
    staticUrls.audio.stamp,
    staticUrls.audio.wind
];

const centerX = window.innerWidth / 2;
const centerY = window.innerHeight / 2;
const exclusionRadius = 150;

function getRandomPosition(element) {
    const section = document.querySelector(".hero");
    const sectionRect = section.getBoundingClientRect();
    const elementWidth = element.clientWidth;
    const elementHeight = element.clientHeight;

    let randomX, randomY;

    do {
        randomX = Math.floor(Math.random() * (sectionRect.width - elementWidth));
        randomY = Math.floor(Math.random() * (sectionRect.height - elementHeight));
    } while (isWithinExclusionZone(randomX, randomY, elementWidth, elementHeight));

    return [randomX, randomY];
}

function isWithinExclusionZone(x, y, width, height) {
    const elementCenterX = x + width / 2;
    const elementCenterY = y + height / 2;
    const distance = Math.sqrt(Math.pow(elementCenterX - centerX, 2) + Math.pow(elementCenterY - centerY, 2));
    return distance < exclusionRadius;
}

function createRandomImage() {
    const section = document.querySelector(".hero");
    const img = document.createElement('img');
    const randomIndex = Math.floor(Math.random() * imagePaths.length);
    img.src = imagePaths[randomIndex];
    img.classList.add('random-image');
    img.style.position = 'absolute';
    img.style.width = '150px';
    section.appendChild(img);

    img.onload = function () {
        const [x, y] = getRandomPosition(img);
        img.style.left = `${x}px`;
        img.style.top = `${y}px`;
    }

    setTimeout(() => {
        img.style.opacity = '0';
        setTimeout(() => {
            img.remove();
        }, 1000);
    }, 2000);
}

function playRandomAudio() {
    const randomSound = audioPaths[Math.floor(Math.random() * audioPaths.length)];
    const audio = new Audio(randomSound);
    audio.play();
}

function showPressText() {
    const prompt = document.getElementById('center-text');
    prompt.textContent = "press";
    isPressVisible = true;
    hasPressed = false;
    startTime = new Date().getTime();

    setTimeout(() => {
        if (isPressVisible) {
            prompt.textContent = "";
            isPressVisible = false;
            if (!hasPressed) {
                if (!flag) {
                    reactionTimeArrayBase.push(-1);
                } else {
                    reactionTimeArrayDistractors.push(-1);
                }
                console.log('No response, appending -1');
            }
        }
    }, Math.random() * 2000 + 1000);
}

function centerReactionTime() {
    const section = document.querySelector(".hero");
    const img = document.createElement('img');
    const prompt = document.createElement('h3');
    img.src = staticUrls.images.redDot;
    section.appendChild(img);
    section.appendChild(prompt);

    img.id = "center-dot";
    img.style.position = 'absolute';
    img.style.left = '50%';
    img.style.top = '50%';
    img.style.width = '100px';
    img.style.transform = 'translate(-50%, -50%)';
    img.style.height = 'auto';

    prompt.id = "center-text";
    prompt.style.position = 'absolute';
    prompt.style.left = '50%';
    prompt.style.top = '48%';
    prompt.style.width = '100px';
    prompt.style.transform = 'translate(-50%, -50%)';
    prompt.style.height = 'auto';
    prompt.textContent = "";

    randomizePressText();
}

function randomizePressText() {
    const interval = Math.random() * (10000 - 3000) + 3000;
    pressTimerId = setTimeout(() => {
        showPressText();
        randomizePressText();
    }, interval);
}

function randomizeImageInterval() {
    const interval = Math.random() * (7000 - 3000) + 3000;
    imageTimerId = setTimeout(() => {
        createRandomImage();
        randomizeImageInterval();
    }, interval);
}

function randomizeAudioInterval() {
    const interval = Math.random() * (7000 - 3000) + 3000;
    audioTimerId = setTimeout(() => {
        playRandomAudio();
        randomizeAudioInterval();
    }, interval);
}

function startTest() {
    centerReactionTime();

    reactionTimeArrayBase = [];
    reactionTimeArrayDistractors = [];

    // Start the baseline test (1 minute without distractions)
    testID = setTimeout(() => {
        console.log("Baseline test complete.");
        flag = true;
        // Start the test with distractions
        randomizeImageInterval();
        randomizeAudioInterval();
        setTimeout(() => {
            concludedFlag = true;
            endTest();
            console.log("Test with distractions complete.");

        }, 60000); // 1 minute with distractions
    }, 60000); // 1 minute for baseline
}

function endTest() {
    clearTimeout(imageTimerId);
    clearTimeout(audioTimerId);
    clearTimeout(pressTimerId);
    clearTimeout(testID)
    const img = document.getElementById('center-dot');
    const prompt = document.getElementById('center-text');
    if (img) img.remove();
    if (prompt) prompt.remove();
    sessionStorage.setItem('reactionTimeBase', JSON.stringify(reactionTimeArrayBase));
    sessionStorage.setItem('reactionTimeDistractors', JSON.stringify(reactionTimeArrayDistractors));
    console.log('Reaction times recorded (in ms):', reactionTimeArrayBase, reactionTimeArrayDistractors);

    if (concludedFlag) {
        concludedFlag = !concludedFlag;
        const event = new CustomEvent('endTest', { detail: "concluded" });
        document.dispatchEvent(event);
    }

}

document.addEventListener('mousedown', (event) => {
    if (event.button === 0 && isPressVisible) {
        reactionTime = new Date().getTime() - startTime;
        document.getElementById('center-text').textContent = "";
        isPressVisible = false;
        hasPressed = true;
        if (!flag) {
            reactionTimeArrayBase.push(reactionTime);
        } else {
            reactionTimeArrayDistractors.push(reactionTime);
        }
        console.log('Reaction Time:', reactionTime, 'ms');
    }
});

function startCalibration() {
    const section = document.querySelector(".hero");
    const img = document.createElement('img');
    img.src = staticUrls.images.redDot;
    section.appendChild(img);

    img.id = "center-dot";
    img.style.position = 'absolute';
    img.style.left = '50%';
    img.style.top = '50%';
    img.style.width = '100px';
    img.style.transform = 'translate(-50%, -50%)';
    img.style.height = 'auto';
}

function endCalibration() {
    const img = document.getElementById('center-dot');
    img.remove();
}

