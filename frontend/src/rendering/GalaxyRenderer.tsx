import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Edge, Paper } from '../types/scholar';

export function GalaxyRenderer({
  papers,
  edges,
  onSelect,
  highlights,
}: {
  papers: Paper[];
  edges: Edge[];
  onSelect: (p: Paper) => void;
  highlights: string[];
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      60,
      mountRef.current.offsetWidth / mountRef.current.offsetHeight,
      0.1,
      1000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.offsetWidth, mountRef.current.offsetHeight);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const idToPos = new Map<string, [number, number, number]>();
    for (const p of papers) idToPos.set(p.id, p.pos);

    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(5000 * 3);
    for (let i = 0; i < 15000; i++) starPos[i] = (Math.random() - 0.5) * 100;
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.02, transparent: true, opacity: 0.4 });
    scene.add(new THREE.Points(starGeo, starMat));

    const edgePositions: number[] = [];
    const edgeColors: number[] = [];
    for (const e of edges) {
      const a = idToPos.get(e.source);
      const b = idToPos.get(e.target);
      if (!a || !b) continue;
      edgePositions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      const base = e.type === 'bridge' ? '#f472b6' : '#60a5fa';
      const c = new THREE.Color(base).multiplyScalar(Math.min(1, 0.35 + e.weight * 0.12));
      edgeColors.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }

    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
    edgeGeo.setAttribute('color', new THREE.Float32BufferAttribute(edgeColors, 3));
    const edgeMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
    });
    const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
    scene.add(edgeLines);

    const sphereGeo = new THREE.SphereGeometry(0.14, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial();
    const mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, papers.length);
    const dummy = new THREE.Object3D();

    scene.add(mesh);

    camera.position.set(0, 0, 15);

    const highlightSet = new Set(highlights);

    const flyTo = (target: THREE.Vector3) => {
      const startCam = camera.position.clone();
      const startTgt = controls.target.clone();
      const endTgt = target.clone();
      const endCam = target.clone().add(new THREE.Vector3(0, 0, 6));
      const start = performance.now();
      const durationMs = 550;

      const step = (now: number) => {
        const t = Math.min(1, (now - start) / durationMs);
        const k = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        camera.position.lerpVectors(startCam, endCam, k);
        controls.target.lerpVectors(startTgt, endTgt, k);
        controls.update();
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };

    const onPointerDown = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(mesh);
      if (intersects.length > 0) {
        const id = intersects[0].instanceId;
        if (id !== undefined) {
          const p = papers[id];
          onSelect(p);
          flyTo(new THREE.Vector3(p.pos[0], p.pos[1], p.pos[2]));
        }
      }
    };

    renderer.domElement.addEventListener('mousedown', onPointerDown);

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();

      const t = performance.now() / 1000;
      for (let i = 0; i < papers.length; i++) {
        const p = papers[i];
        const isHi = highlightSet.has(p.id);
        const s = isHi ? 1 + 0.35 * Math.sin(t * 5 + i * 0.2) : 1;
        dummy.position.set(p.pos[0], p.pos[1], p.pos[2]);
        dummy.scale.setScalar(s);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        mesh.setColorAt(i, new THREE.Color(isHi ? '#fbbf24' : p.color));
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.offsetWidth / mountRef.current.offsetHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.offsetWidth, mountRef.current.offsetHeight);
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('mousedown', onPointerDown);
      cancelAnimationFrame(raf);
      edgeGeo.dispose();
      edgeMat.dispose();
      sphereGeo.dispose();
      sphereMat.dispose();
      mesh.dispose();
      starGeo.dispose();
      starMat.dispose();
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [papers, edges, highlights, onSelect]);

  return <div ref={mountRef} className="w-full h-full" />;
}
