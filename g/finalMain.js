   
'use strict';

const ROOM_SIZE = 50;
const ROOM_MIN = -ROOM_SIZE / 2;
const ROOM_MAX = ROOM_SIZE / 2;
const WALL_PADDING = 0.5;
let soccerBallTexture;
let mySoccerBall = null;
let myCouch = null;
let couchTexture;
let tableTop = null;
let tableLegs = [];
const lightPos = [0.0, ROOM_MAX - 5.0, 0.0];
const lightColor = [1.0, 1.0, 1.0];
let pointLightPosition = [0, ROOM_MAX - 5, 0]; // Near the ceiling
let pointLightColor = [2.0,0.0,0.0]; // Warm white light
let pointLightIntensity = 1.0;
let pointLightVAO = null;
let lightSphere = null;

class Couch {
    constructor() {
        this.points = [];
        this.uv = [];
        this.normals = [];
        this.indices = [];
        this.setupCouch();
    }

    setupCouch() {
        const seatLength = 10.0;
        const seatWidth = 2.0;
        const seatHeight = 2;
        const cushionHeight = 0.4;
        const backHeight = 1.5;

        this.addBox(-seatWidth/2, -seatHeight/2, -seatLength/2, 
                   seatWidth/2, -seatHeight/2 + cushionHeight, seatLength/2);
        this.addBox(-seatWidth/2, -seatHeight/2 + cushionHeight, -seatLength/2, 
                   -seatWidth/2 + 0.4, -seatHeight/2 + cushionHeight + backHeight, seatLength/2);
        this.addBox(-seatWidth/2, -seatHeight/2 + cushionHeight, -seatLength/2, 
                   seatWidth/2, -seatHeight/2 + cushionHeight + 0.8, -seatLength/2 + 0.5);
        this.addBox(-seatWidth/2, -seatHeight/2 + cushionHeight, seatLength/2 - 0.5, 
                   seatWidth/2, -seatHeight/2 + cushionHeight + 0.8, seatLength/2);
    }

    addBox(x1, y1, z1, x2, y2, z2) {
        const baseIndex = this.points.length / 3;
        const vertices = [
            [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
            [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]
        ];
        const faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4],
            [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]
        ];
        const normals = [
            [0, 0, -1], [0, 0, 1], [-1, 0, 0],
            [1, 0, 0], [0, -1, 0], [0, 1, 0]
        ];
        
        for (let i = 0; i < faces.length; i++) {
            const face = faces[i];
            const normal = normals[i];
            for (const vi of face) {
                this.points.push(...vertices[vi]);
                this.normals.push(...normal);
            }
            this.uv.push(0, 0, 1, 0, 1, 1, 0, 1);
            const newIndices = [
                baseIndex + i*4, baseIndex + i*4 + 1, baseIndex + i*4 + 2,
                baseIndex + i*4, baseIndex + i*4 + 2, baseIndex + i*4 + 3
            ];
            this.indices.push(...newIndices);
        }
    }
}

class Room {
    constructor(size) {
        this.size = size;
        this.points = [];
        this.uv = [];
        this.normals = [];
        this.indices = [];
        this.setupRoom();
    }

    setupRoom() {
        const s = this.size / 2;
        this.addFace([-s, -s, -s], [s, -s, -s], [s, -s, s], [-s, -s, s],
            [0, 0], [1, 0], [1, 1], [0, 1], [0, 1, 0]);
        this.addFace([-s, s, s], [s, s, s], [s, s, -s], [-s, s, -s],
            [0, 0], [1, 0], [1, 1], [0, 1], [0, -1, 0]);
        this.addFace([-s, -s, -s], [-s, s, -s], [s, s, -s], [s, -s, -s],
            [0, 0], [0, 1], [1, 1], [1, 0], [0, 0, 1]);
        this.addFace([s, -s, s], [s, s, s], [-s, s, s], [-s, -s, s],
            [0, 0], [0, 1], [1, 1], [1, 0], [0, 0, -1]);
        this.addFace([-s, -s, s], [-s, s, s], [-s, s, -s], [-s, -s, -s],
            [0, 0], [0, 1], [1, 1], [1, 0], [1, 0, 0]);
        this.addFace([s, -s, -s], [s, s, -s], [s, s, s], [s, -s, s],
            [0, 0], [0, 1], [1, 1], [1, 0], [-1, 0, 0]);
    }

    addFace(v0, v1, v2, v3, uv0, uv1, uv2, uv3, normal) {
        const baseIndex = this.points.length / 3;
        this.points.push(...v0, ...v1, ...v2, ...v3);
        this.uv.push(...uv0, ...uv1, ...uv2, ...uv3);
        this.normals.push(...normal, ...normal, ...normal, ...normal);
        this.indices.push(baseIndex, baseIndex+1, baseIndex+2, baseIndex, baseIndex+2, baseIndex+3);
    }
}

let gl;
let customTexture;
let cubeProgram;
let myCube = null;
const mat4 = glMatrix.mat4;

let cameraPos = [0, 0, 3];
let cameraFront = [0, 0, -1];
let cameraUp = [0, 1, 0];
let cameraSpeed = 0.5;
const mouseSensitivity = 0.6;
let yaw = -90;
let pitch = 0;
let lastX = 0;
let lastY = 0;
let firstMouse = true;
const keysPressed = {
    'w': false,
    'a': false,
    's': false,
    'd': false,
    'q': false,
    'e': false
};

function createTable() {
    const tableRadius = 10;
    const tableThickness = 0.2;
    const legCount = 4;
    const legRadius = 100;
    const legHeight = 3;
    const legPositionRadius = 4;

    tableTop = new Cylinder(100.0, tableThickness, 32, 1);
    tableTop.VAO = bindVAO(tableTop, cubeProgram);
    tableTop.modelMatrix = mat4.create();
    mat4.translate(tableTop.modelMatrix, tableTop.modelMatrix, [0, -ROOM_SIZE/1.9 + 2.5, 0]);
    mat4.scale(tableTop.modelMatrix, tableTop.modelMatrix, [tableRadius, 1, tableRadius]);

    for (let i = 0; i < legCount; i++) {
        let leg = new Cylinder(legRadius, 12, 32, 1);
        leg.VAO = bindVAO(leg, cubeProgram);
        leg.modelMatrix = mat4.create();
        const angle = (i / legCount) * Math.PI * 2;
        const xPos = Math.cos(angle) * legPositionRadius;
        const zPos = Math.sin(angle) * legPositionRadius;
        mat4.translate(leg.modelMatrix, leg.modelMatrix, [xPos, -ROOM_SIZE/1.9 + legHeight/2, zPos]);
        tableLegs.push(leg);
    }
}

function setUpTextures() {
    soccerBallTexture = gl.createTexture();
    const soccerBallImage = new Image();
    soccerBallImage.src = 'football.jpg';
    soccerBallImage.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, soccerBallTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, soccerBallImage);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.bindTexture(gl.TEXTURE_2D, null);
        draw();
    };

    customTexture = gl.createTexture();
    const roomImage = new Image();
    roomImage.src = 'bricks.jpg';
    roomImage.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, customTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, roomImage);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.bindTexture(gl.TEXTURE_2D, null);
    };

    couchTexture = gl.createTexture();
    const couchImage = new Image();
    couchImage.src = 'fabric.jpg';
    couchImage.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, couchTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, couchImage);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.bindTexture(gl.TEXTURE_2D, null);
        draw();
    };
}

function getShader(id) {
    const script = document.getElementById(id);
    const shaderString = script.text.trim();
    let shader;
    if (script.type === 'x-shader/x-vertex') {
        shader = gl.createShader(gl.VERTEX_SHADER);
    }
    else if (script.type === 'x-shader/x-fragment') {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    }
    else {
        return null;
    }

    gl.shaderSource(shader, shaderString);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Compiling shader " + id + " " + gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}

function initProgram(vertexid, fragmentid) {
    const vertexShader = getShader(vertexid);
    const fragmentShader = getShader(fragmentid);
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Could not initialize shaders');
    }

    gl.useProgram(program);
    program.aVertexPosition = gl.getAttribLocation(program, 'aVertexPosition');
    program.aUV = gl.getAttribLocation(program, 'aUV');
    program.aNormal = gl.getAttribLocation(program, 'aNormal');
    program.uTheTexture = gl.getUniformLocation(program, 'theTexture');
    program.uModelMatrix = gl.getUniformLocation(program, 'uModelMatrix');
    program.uModelViewMatrix = gl.getUniformLocation(program, 'uModelViewMatrix');
    program.uProjectionMatrix = gl.getUniformLocation(program, 'uProjectionMatrix');
    program.uNormalMatrix = gl.getUniformLocation(program, 'uNormalMatrix');
    program.uLightPos = gl.getUniformLocation(program, 'lightPos');
    program.uLightColor = gl.getUniformLocation(program, 'lightColor');
    program.uViewPos = gl.getUniformLocation(program, 'viewPos');
    program.uPointLightPosition = gl.getUniformLocation(program, 'pointLightPosition');
    program.uPointLightColor = gl.getUniformLocation(program, 'pointLightColor');
    program.uPointLightIntensity = gl.getUniformLocation(program, 'pointLightIntensity');
    
    
    return program;
}

function createShapes() {
    createTable();
    myCube = new Room(ROOM_SIZE);
    myCube.VAO = bindVAO(myCube, cubeProgram);
    mySoccerBall = new Sphere(15.0, 100, 100);
    mySoccerBall.VAO = bindVAO(mySoccerBall, cubeProgram);
    mySoccerBall.modelMatrix = mat4.create();
    mat4.translate(mySoccerBall.modelMatrix, mySoccerBall.modelMatrix, [-ROOM_MAX + 15, -ROOM_SIZE/2+0.75, 0]);
    mat4.scale(mySoccerBall.modelMatrix, mySoccerBall.modelMatrix, [1.5, 1.5, 1.5]);

    myCouch = new Couch();
    myCouch.VAO = bindVAO(myCouch, cubeProgram);
    myCouch.modelMatrix = mat4.create();
    mat4.translate(myCouch.modelMatrix, myCouch.modelMatrix, [-ROOM_MAX + 2, -ROOM_SIZE/2 + 2, 0]);
    mat4.rotateY(myCouch.modelMatrix, myCouch.modelMatrix, 4*Math.PI/2);
    const scaleFactor = 4;
    mat4.scale(myCouch.modelMatrix, myCouch.modelMatrix, [scaleFactor, scaleFactor, scaleFactor]);
    mat4.translate(myCouch.modelMatrix, myCouch.modelMatrix, [1,0.5, 0]);
    lightSphere = new Sphere(15.0, 16, 16);
    lightSphere.VAO = bindVAO(lightSphere, cubeProgram);
    lightSphere.modelMatrix = mat4.create();
    mat4.translate(lightSphere.modelMatrix, lightSphere.modelMatrix, pointLightPosition);
    mat4.scale(lightSphere.modelMatrix, lightSphere.modelMatrix, [0.3, 0.3, 0.3]);
}

function bindVAO(shape, program) {
    let theVAO = gl.createVertexArray();
    gl.bindVertexArray(theVAO);
    
    let myVertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, myVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(shape.points), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(program.aVertexPosition);
    gl.vertexAttribPointer(program.aVertexPosition, 3, gl.FLOAT, false, 0, 0);
    
    let uvBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(shape.uv), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(program.aUV);
    gl.vertexAttribPointer(program.aUV, 2, gl.FLOAT, false, 0, 0);
    
    let normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(shape.normals), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(program.aNormal);
    gl.vertexAttribPointer(program.aNormal, 3, gl.FLOAT, false, 0, 0);
    
    let myIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, myIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(shape.indices), gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    
    return theVAO;
}

function drawCurrentShape() {
    gl.useProgram(cubeProgram);
    
    let viewMatrix = mat4.create();
    let projectionMatrix = mat4.create();
    
    mat4.lookAt(viewMatrix, cameraPos, 
               glMatrix.vec3.add([], cameraPos, cameraFront), 
               cameraUp);
    
    mat4.perspective(projectionMatrix, 
                    45 * Math.PI / 180, 
                    gl.canvas.width / gl.canvas.height, 
                    0.1, 
                    100.0);
    if (lightSphere) {
        gl.disable(gl.DEPTH_TEST); // Make light always visible
        gl.uniform3fv(cubeProgram.uPointLightColor, [1.0, 1.0, 0.0]); // Yellow
        drawObject(lightSphere, viewMatrix, projectionMatrix);
        gl.enable(gl.DEPTH_TEST);
    }
    drawObject(myCube, viewMatrix, projectionMatrix);
    drawObject(myCouch, viewMatrix, projectionMatrix);
    drawObject(mySoccerBall, viewMatrix, projectionMatrix);
    drawObject(tableTop, viewMatrix, projectionMatrix);
    tableLegs.forEach(leg => drawObject(leg, viewMatrix, projectionMatrix));
}

function drawObject(object, viewMatrix, projectionMatrix) {
    let modelViewMatrix = mat4.create();
    let normalMatrix = mat4.create();
    
    mat4.multiply(modelViewMatrix, viewMatrix, object.modelMatrix || mat4.create());
    mat4.invert(normalMatrix, modelViewMatrix);
    mat4.transpose(normalMatrix, normalMatrix);
    
    gl.uniformMatrix4fv(cubeProgram.uModelViewMatrix, false, modelViewMatrix);
    gl.uniformMatrix4fv(cubeProgram.uProjectionMatrix, false, projectionMatrix);
    gl.uniformMatrix4fv(cubeProgram.uModelMatrix, false, object.modelMatrix || mat4.create());
    gl.uniformMatrix4fv(cubeProgram.uNormalMatrix, false, normalMatrix);
    gl.uniform3fv(cubeProgram.uLightPos, lightPos);
    gl.uniform3fv(cubeProgram.uLightColor, lightColor);
    gl.uniform3fv(cubeProgram.uViewPos, cameraPos);
    gl.uniform3fv(cubeProgram.uPointLightPosition, pointLightPosition);
    gl.uniform3fv(cubeProgram.uPointLightColor, pointLightColor);
    gl.uniform1f(cubeProgram.uPointLightIntensity, pointLightIntensity);
    gl.activeTexture(gl.TEXTURE0);
    if (object === myCouch) {
        gl.bindTexture(gl.TEXTURE_2D, couchTexture);
    } else if (object === mySoccerBall) {
        gl.bindTexture(gl.TEXTURE_2D, soccerBallTexture);
    } else {
        gl.bindTexture(gl.TEXTURE_2D, customTexture);
    }
    gl.uniform1i(cubeProgram.uTheTexture, 0);
    
    gl.bindVertexArray(object.VAO);
    gl.drawElements(gl.TRIANGLES, object.indices.length, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
}

function draw() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    drawCurrentShape();
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
}

function updateCameraVectors() {
    let front = [
        Math.cos(glMatrix.glMatrix.toRadian(yaw)) * Math.cos(glMatrix.glMatrix.toRadian(pitch)),
        Math.sin(glMatrix.glMatrix.toRadian(pitch)),
        Math.sin(glMatrix.glMatrix.toRadian(yaw)) * Math.cos(glMatrix.glMatrix.toRadian(pitch))
    ];
    cameraFront = glMatrix.vec3.normalize([], front);
}

function handleMouseMove(event) {
    if (firstMouse) {
        lastX = event.clientX;
        lastY = event.clientY;
        firstMouse = false;
        return;
    }

    let xoffset = event.clientX - lastX;
    let yoffset = lastY - event.clientY;
    lastX = event.clientX;
    lastY = event.clientY;

    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0) pitch = 89.0;
    if (pitch < -89.0) pitch = -89.0;

    updateCameraVectors();
}

function handleKeyDown(event) {
    const key = event.key.toLowerCase();
    keysPressed[key] = true;
}

function handleKeyUp(event) {
    const key = event.key.toLowerCase();
    keysPressed[key] = false;
}

function init() {
    cameraPos=[0.0,0.0,0.0];
    const canvas = document.getElementById('webgl-canvas');
    if (!canvas) {
        console.error(`There is no canvas with id ${'webgl-canvas'} on this page.`);
        return null;
    }

    gl = canvas.getContext('webgl2');
    if (!gl) {
        console.error(`There is no WebGL 2.0 context`);
        return null;
    }
      
    gl.enable(gl.DEPTH_TEST);
    gl.frontFace(gl.CCW);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.depthFunc(gl.LEQUAL);
    gl.clearDepth(1.0);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      
    cubeProgram = initProgram('sphereMap-V', 'sphereMap-F');
    createShapes();
    setUpTextures();
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    canvas.addEventListener('mousemove', handleMouseMove);
    
    canvas.requestPointerLock = canvas.requestPointerLock || 
                              canvas.mozRequestPointerLock || 
                              canvas.webkitRequestPointerLock;
    canvas.onclick = function() {
        canvas.requestPointerLock();
    };

    document.addEventListener('pointerlockchange', lockChangeAlert, false);
    document.addEventListener('mozpointerlockchange', lockChangeAlert, false);
    document.addEventListener('webkitpointerlockchange', lockChangeAlert, false);

    function lockChangeAlert() {
        if (document.pointerLockElement === canvas || 
            document.mozPointerLockElement === canvas || 
            document.webkitPointerLockElement === canvas) {
            console.log('Pointer locked');
        } else {
            console.log('Pointer unlocked');
        }
    }

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    
    const velocity = glMatrix.vec3.scale([], cameraFront, cameraSpeed);
    const right = glMatrix.vec3.normalize([], 
        glMatrix.vec3.cross([], cameraFront, cameraUp));

    let newPos = [...cameraPos];
    
    if (keysPressed['w']) glMatrix.vec3.add(newPos, newPos, velocity);
    if (keysPressed['s']) glMatrix.vec3.sub(newPos, newPos, velocity);
    if (keysPressed['a']) glMatrix.vec3.sub(newPos, newPos, 
        glMatrix.vec3.scale([], right, cameraSpeed));
    if (keysPressed['d']) glMatrix.vec3.add(newPos, newPos, 
        glMatrix.vec3.scale([], right, cameraSpeed));
    if (keysPressed['q']) newPos[1] -= cameraSpeed;
    if (keysPressed['e']) newPos[1] += cameraSpeed;
    pointLightPosition[0] = Math.sin(Date.now() * 0.001) * 3;
    pointLightPosition[2] = Math.cos(Date.now() * 0.001) * 3;
    
    // Update light sphere position
    if (lightSphere) {
        mat4.identity(lightSphere.modelMatrix);
        mat4.translate(lightSphere.modelMatrix, lightSphere.modelMatrix, pointLightPosition);
        mat4.scale(lightSphere.modelMatrix, lightSphere.modelMatrix, [0.3, 0.3, 0.3]);
    }
    // Clamp position to room boundaries
    // newPos[0] = Math.max(ROOM_MIN + WALL_PADDING, Math.min(ROOM_MAX - WALL_PADDING, newPos[0]));
    // newPos[1] = Math.max(ROOM_MIN + WALL_PADDING, Math.min(ROOM_MAX - WALL_PADDING, newPos[1]));
    // newPos[2] = Math.max(ROOM_MIN + WALL_PADDING, Math.min(ROOM_MAX - WALL_PADDING, newPos[2]));
    cameraPos = newPos;
    draw();
}

window.onload = init;