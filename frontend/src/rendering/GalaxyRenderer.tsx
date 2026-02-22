import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Line, Stars } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import type { Edge, Paper } from '../types/scholar';

// --- 连线组件：科技感渐变与交互 ---
function EdgeConnection({ from, to }: { from: Paper; to: Paper }) {
  const [hovered, setHover] = useState(false);
  
  // 预计算颜色对象
  const colors = useMemo(() => {
    return [new THREE.Color(from.color), new THREE.Color(to.color)];
  }, [from.color, to.color]);

  return (
    <group>
      {/* 1. 可见线：极简渐变，悬停时高亮并发光 */}
      <Line
        points={[from.pos, to.pos]}
        vertexColors={colors}
        lineWidth={hovered ? 3 : 1} // 悬停变粗
        transparent
        opacity={hovered ? 1.0 : 0.2} // 平时低透明度，悬停完全不透明
        depthWrite={false}
        toneMapped={false} // 允许颜色溢出产生辉光
      />
      
      {/* 2. 交互热区：不可见的粗线，用于捕捉鼠标事件 */}
      <Line
        points={[from.pos, to.pos]}
        lineWidth={12} // 增加点击范围
        transparent
        opacity={0} // 完全透明
        onPointerOver={(e) => {
          e.stopPropagation(); // 防止穿透
          setHover(true);
        }}
        onPointerOut={() => setHover(false)}
      />
    </group>
  );
}

// --- 核心渲染组件：渐变发光星球 ---
function PaperNode({ 
  paper, 
  isHighlighted, 
  onSelect,
  hideLabels
}: { 
  paper: Paper; 
  isHighlighted: boolean; 
  onSelect: (p: Paper) => void;
  hideLabels?: boolean;
}) {
  const [hovered, setHover] = useState(false);
  const position = useMemo(() => new THREE.Vector3(...paper.pos), [paper.pos]);

  // 根据选中或悬停状态计算颜色和强度
  const baseColor = isHighlighted ? '#fbbf24' : paper.color;
  const glowIntensity = hovered || isHighlighted ? 2.8 : 0.8;

  return (
    <group position={position}>
      {/* 1. 核心球体：使用物理材质并开启自发光 */}
      <mesh 
        onClick={(e) => {
          e.stopPropagation();
          onSelect(paper);
        }}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        <sphereGeometry args={[0.27, 32, 32]} />
        <meshStandardMaterial 
          color={baseColor}
          emissive={baseColor}
          emissiveIntensity={glowIntensity}
          roughness={0.2}
          metalness={0.8}
          toneMapped={false}
        />
      </mesh>

      {/* 2. 渐变外壳：通过增加一个略大的半透明球体模拟大气渐变层 */}
      <mesh scale={[1.2, 1.2, 1.2]}>
        <sphereGeometry args={[0.27, 32, 32]} />
        <meshBasicMaterial 
          color={baseColor}
          transparent
          opacity={0.15}
          blending={THREE.AdditiveBlending}
          side={THREE.BackSide}
        />
      </mesh>

      {/* 3. 科技感文字标签 */}
      {!hideLabels && (
        <Html distanceFactor={15} position={[0, 0.9, 0]} center>
          <div style={{
            pointerEvents: 'none',
            background: hovered ? 'rgba(2, 6, 12, 0.9)' : 'rgba(2, 6, 12, 0.4)',
            color: 'white',
            padding: '4px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 500,
            border: `1px solid ${baseColor}`,
            whiteSpace: 'nowrap',
            boxShadow: hovered ? `0 0 20px ${baseColor}` : `0 0 10px ${baseColor}33`,
            backdropFilter: 'blur(6px)',
            transition: 'all 0.4s cubic-bezier(0.23, 1, 0.32, 1)',
            opacity: hovered || isHighlighted ? 1 : 0.7,
          }}>
            {paper.displayTitle || paper.title}
          </div>
        </Html>
      )}
    </group>
  );
}

// --- 主渲染组件 ---
export function GalaxyRenderer({
  papers,
  edges,
  onSelect,
  highlights,
  hideLabels,
  focusTarget,
}: {
  papers: Paper[];
  edges: Edge[];
  onSelect: (p: Paper) => void;
  highlights: string[];
  hideLabels?: boolean;
  focusTarget?: Paper | null;
}) {
  const highlightSet = useMemo(() => new Set(highlights), [highlights]);
  // Camera & controls refs
  const controlsRef = useRef<any>(null);

  // Camera animator must run inside the Canvas render context, so define a small component
  function CameraAnimator({ focus }: { focus?: Paper | null }) {
    const { camera } = useThree();
    const animRef = useRef<number | null>(null);
    const startRef = useRef<{ from: THREE.Vector3; to: THREE.Vector3; targetFrom: THREE.Vector3; targetTo: THREE.Vector3; startTime: number; duration: number } | null>(null);

    useEffect(() => {
      if (!focus) return;
      const targetPos = new THREE.Vector3(...focus.pos);
      const offset = new THREE.Vector3(0, 0, 18 + (focus.size || 0));
      const desiredCam = targetPos.clone().add(offset);

      const from = camera.position.clone();
      const to = desiredCam;
      const targetFrom = controlsRef.current ? controlsRef.current.target.clone() : new THREE.Vector3(0, 0, 0);
      const targetTo = targetPos.clone();
      const duration = 600;
      startRef.current = { from, to, targetFrom, targetTo, startTime: performance.now(), duration };

      if (animRef.current) cancelAnimationFrame(animRef.current);

      const step = () => {
        if (!startRef.current) return;
        const now = performance.now();
        const t = Math.min(1, (now - startRef.current.startTime) / startRef.current.duration);
        const ease = t * (2 - t);
        camera.position.lerpVectors(startRef.current.from, startRef.current.to, ease);
        if (controlsRef.current) {
          const newTarget = startRef.current.targetFrom.clone().lerp(startRef.current.targetTo, ease);
          controlsRef.current.target.copy(newTarget);
          controlsRef.current.update();
        }

        if (t < 1) {
          animRef.current = requestAnimationFrame(step);
        } else {
          startRef.current = null;
          animRef.current = null;
        }
      };

      animRef.current = requestAnimationFrame(step);
      return () => {
        if (animRef.current) cancelAnimationFrame(animRef.current);
        animRef.current = null;
      };
    }, [focus]);

    return null;
  }

  return (
    <div style={{ width: '100%', height: '100%', background: '#010204', position: 'relative' }}>
      <Canvas 
        camera={{ position: [0, 0, 60], fov: 50 }}
        gl={{ 
          antialias: true, 
          toneMapping: THREE.ReinhardToneMapping,
          powerPreference: "high-performance" 
        }}
      >
        <color attach="background" args={['#010204']} />
        
        {/* 背景氛围 */}
        <Stars radius={150} depth={50} count={5000} factor={4} saturation={0.5} fade speed={1.5} />
        <ambientLight intensity={0.2} />
        <pointLight position={[20, 20, 20]} intensity={1.5} color="#ffffff" />

        {/* Camera animator listens to focusTarget and performs fly-to */}
        <CameraAnimator focus={focusTarget} />

        {/* 渲染连接线：使用自定义组件实现科技感交互 */}
        {edges.map((edge, i) => {
          const from = papers.find(p => p.id === edge.source);
          const to = papers.find(p => p.id === edge.target);
          if (!from || !to) return null;
          
          return (
            <EdgeConnection 
              key={`edge-${i}`}
              from={from}
              to={to}
            />
          );
        })}

        {/* 渲染节点 */}
        {papers.map((paper) => (
          <PaperNode 
            key={paper.id} 
            paper={paper} 
            isHighlighted={highlightSet.has(paper.id)}
            onSelect={onSelect}
            hideLabels={hideLabels}
          />
        ))}

        {/* 交互控制 */}
        <OrbitControls 
          ref={controlsRef}
          enableDamping 
          dampingFactor={0.05} 
          rotateSpeed={0.7}
          minDistance={10}
          maxDistance={200}
        />

        {/* 关键渲染步骤：辉光后期处理 */}
        <EffectComposer disableNormalPass>
          <Bloom 
            luminanceThreshold={0.15} 
            intensity={1.8} 
            mipmapBlur 
            radius={0.3} 
          />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
