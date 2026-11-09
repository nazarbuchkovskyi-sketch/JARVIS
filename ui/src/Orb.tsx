import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { JarvisState, Levels } from "./useJarvis";

// Monochrome: bone particles, only brightness changes with the state.
const PALETTE: Record<JarvisState, [number, number, number]> = {
  offline: [0.28, 0.28, 0.27],
  booting: [0.45, 0.45, 0.44],
  idle: [0.62, 0.62, 0.6],
  listening: [0.95, 0.95, 0.93],
  transcribing: [0.85, 0.85, 0.83],
  thinking: [0.75, 0.75, 0.73],
  speaking: [1.0, 1.0, 0.98],
  approval: [0.85, 0.47, 0.34],
};

// Classic 3D simplex noise (Ashima / Stefan Gustavson), used to ripple the sphere.
const NOISE = /* glsl */ `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z); vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`;

const VERTEX = /* glsl */ `
uniform float uTime; uniform float uLevel; uniform float uAgitation; uniform float uPixelRatio;
attribute float aSeed;
varying float vGlow;
${NOISE}
void main(){
  vec3 dir = normalize(position);
  float n = snoise(dir * 1.8 + vec3(0.0, uTime * (0.25 + uAgitation), 0.0));
  float ripple = snoise(dir * 4.0 - vec3(uTime * 0.9));
  float disp = n * (0.07 + uLevel * 0.55) + ripple * uLevel * 0.18;
  vec3 pos = dir * (1.0 + disp);
  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  float twinkle = 0.6 + 0.4 * sin(uTime * 2.0 + aSeed * 40.0);
  vGlow = clamp(0.35 + disp * 2.2 + uLevel * 0.8, 0.15, 1.6) * twinkle;
  gl_PointSize = (1.4 + aSeed * 1.6 + uLevel * 2.0) * uPixelRatio * (4.0 / -mv.z);
}`;

const FRAGMENT = /* glsl */ `
uniform vec3 uColor;
varying float vGlow;
void main(){
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5) discard;
  float a = smoothstep(0.5, 0.0, d);
  gl_FragColor = vec4(uColor * (0.6 + vGlow), a * vGlow);
}`;

function fibonacciSphere(count: number): Float32Array {
  const pts = new Float32Array(count * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = golden * i;
    pts[i * 3] = Math.cos(theta) * r;
    pts[i * 3 + 1] = y;
    pts[i * 3 + 2] = Math.sin(theta) * r;
  }
  return pts;
}

function ring(radius: number, segments: number, dashed: boolean, color: THREE.Color) {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a) * radius, Math.sin(a) * radius, 0));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.06, gapSize: 0.05, transparent: true, opacity: 0.32 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.16 });
  const line = new THREE.Line(geo, mat);
  if (dashed) line.computeLineDistances();
  return line;
}

export function Orb({ state, levels }: { state: JarvisState; levels: React.MutableRefObject<Levels> }) {
  const host = useRef<HTMLDivElement>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    const el = host.current!;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    camera.position.set(0, 0, 4.4);

    const count = 6000;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(fibonacciSphere(count), 3));
    const seeds = new Float32Array(count);
    for (let i = 0; i < count; i++) seeds[i] = Math.random();
    geo.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));

    const uniforms = {
      uTime: { value: 0 },
      uLevel: { value: 0 },
      uAgitation: { value: 0 },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uColor: { value: new THREE.Color(...PALETTE.idle) },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const sphere = new THREE.Points(geo, mat);
    scene.add(sphere);

    const rings = new THREE.Group();
    const ringColor = new THREE.Color(...PALETTE.idle);
    const r1 = ring(1.42, 256, true, ringColor);
    const r2 = ring(1.6, 256, false, ringColor);
    const r3 = ring(1.25, 180, true, ringColor);
    r1.rotation.x = 1.25;
    r2.rotation.x = 1.45;
    r2.rotation.y = 0.2;
    r3.rotation.y = 1.1;
    rings.add(r1, r2, r3);
    scene.add(rings);

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.position.z = camera.aspect < 1 ? 4.4 / camera.aspect : 4.4;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const target = new THREE.Color();
    let level = 0;
    let agitation = 0;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const s = stateRef.current;
      const raw = s === "speaking" ? levels.current.out * 9 : s === "listening" ? levels.current.mic * 14 : 0;
      const want = Math.min(raw, 1);
      level += (want - level) * (want > level ? 0.45 : 0.12);
      const wantAgitation = s === "thinking" ? 1.2 : s === "transcribing" ? 0.7 : s === "speaking" ? 0.4 : 0.05;
      agitation += (wantAgitation - agitation) * 0.05;

      target.setRGB(...PALETTE[s]);
      uniforms.uColor.value.lerp(target, 0.08);
      ringColor.lerp(target, 0.08);
      for (const r of rings.children) ((r as THREE.Line).material as THREE.LineBasicMaterial).color.copy(ringColor);

      const speed = reduced ? 0.2 : 1;
      uniforms.uTime.value += dt * speed;
      uniforms.uLevel.value = reduced ? level * 0.4 : level;
      uniforms.uAgitation.value = agitation;
      sphere.rotation.y += dt * (0.08 + agitation * 0.5) * speed;
      r1.rotation.z += dt * (0.25 + agitation) * speed;
      r2.rotation.z -= dt * (0.12 + agitation * 0.6) * speed;
      r3.rotation.z += dt * (0.4 + agitation * 1.5) * speed;
      const scale = 1 + level * 0.06;
      rings.scale.setScalar(scale);

      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      geo.dispose();
      mat.dispose();
      rings.children.forEach((r) => {
        (r as THREE.Line).geometry.dispose();
        ((r as THREE.Line).material as THREE.Material).dispose();
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [levels]);

  return <div className="orb" ref={host} aria-hidden="true" />;
}
