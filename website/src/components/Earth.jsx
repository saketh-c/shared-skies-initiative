import { Suspense, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Sphere, useTexture, Stars } from '@react-three/drei'
import * as THREE from 'three'

const EARTH_TEXTURE = 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg'

function Globe() {
  const groupRef = useRef()
  const texture = useTexture(EARTH_TEXTURE)

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.055
    }
  })

  return (
    <group ref={groupRef} rotation={[0.28, 1.75, 0]}>
      {/* Main sphere */}
      <Sphere args={[2.5, 64, 64]}>
        <meshPhongMaterial
          map={texture}
          specular={new THREE.Color(0x222233)}
          shininess={12}
        />
      </Sphere>
      {/* Outer atmosphere */}
      <Sphere args={[2.68, 32, 32]}>
        <meshBasicMaterial
          color="#5588ff"
          transparent
          opacity={0.055}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </Sphere>
      {/* Inner atmosphere rim */}
      <Sphere args={[2.58, 32, 32]}>
        <meshBasicMaterial
          color="#88aaff"
          transparent
          opacity={0.035}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </Sphere>
    </group>
  )
}

function EarthFallback() {
  const mesh = useRef()
  useFrame((_, dt) => { if (mesh.current) mesh.current.rotation.y += dt * 0.055 })
  return (
    <mesh ref={mesh} rotation={[0.28, 0, 0]}>
      <sphereGeometry args={[2.5, 32, 32]} />
      <meshPhongMaterial color="#2a5aaa" shininess={10} />
    </mesh>
  )
}

export default function Earth() {
  return (
    <Canvas
      style={{ width: '100%', height: '100%', background: 'transparent' }}
      camera={{ position: [0, 0, 7.2], fov: 40 }}
      gl={{ antialias: true, alpha: true }}
    >
      <Stars radius={280} depth={55} count={1800} factor={4} fade saturation={0} />
      <ambientLight intensity={0.2} />
      <directionalLight position={[8, 4, 6]} intensity={1.55} color="#fff8f2" />
      <pointLight position={[-10, -5, -10]} intensity={0.08} color="#4488ff" />
      <Suspense fallback={<EarthFallback />}>
        <Globe />
      </Suspense>
    </Canvas>
  )
}
