import { useRef, useEffect, useState } from 'react'
import { motion, useInView } from 'framer-motion'

function useCountUp(target, active) {
  const [count, setCount] = useState(0)
  useEffect(() => {
    if (!active) return
    const duration = 1800
    const start = Date.now()
    const timer = setInterval(() => {
      const t = Math.min((Date.now() - start) / duration, 1)
      setCount(Math.round((1 - Math.pow(1 - t, 3)) * target))
      if (t >= 1) clearInterval(timer)
    }, 16)
    return () => clearInterval(timer)
  }, [active, target])
  return count
}

const STATS = [
  { raw: 6500, suffix: '+', label: 'Dollars Raised', desc: 'We are actively fundraising, with our campaign launched in September 2025.' },
  { raw: 10, suffix: '+', label: 'Partnerships', desc: 'We work with local organizations, universities, schools, and businesses.' },
  { raw: 20, suffix: '+', label: 'Volunteers', desc: 'We have a team of over 20 dedicated students from across the United States.' },
]

function StatCard({ stat, index }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.4 })
  const count = useCountUp(stat.raw, inView)

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 28 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ type: 'spring', stiffness: 55, damping: 20, delay: index * 0.1 }}
      whileHover={{ y: -6, boxShadow: '0 20px 48px rgba(0,0,0,0.18)' }}
      style={{
        background: 'white',
        borderRadius: '8px',
        padding: '44px 40px',
        minHeight: '270px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}
    >
      <div>
        <p style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '60px', fontWeight: 600, color: '#2d52b8', lineHeight: 1, marginBottom: '8px' }}>
          {count.toLocaleString()}{stat.suffix}
        </p>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '16px', fontWeight: 600, color: '#2d52b8' }}>
          {stat.label}
        </p>
      </div>
      <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '13px', color: '#555', lineHeight: 1.65, marginTop: '20px' }}>
        {stat.desc}
      </p>
    </motion.div>
  )
}

export default function Facts() {
  const headRef = useRef(null)
  const headInView = useInView(headRef, { once: true, amount: 0.4 })

  return (
    <section style={{ background: '#1e3d6e', padding: '80px 68px 100px' }}>
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.15)', paddingTop: '56px' }}>
        <motion.h2
          ref={headRef}
          initial={{ opacity: 0, y: 20 }}
          animate={headInView ? { opacity: 1, y: 0 } : {}}
          transition={{ type: 'spring', stiffness: 55, damping: 20 }}
          style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '30px', fontWeight: 500, color: 'white', marginBottom: '36px', letterSpacing: '-0.01em' }}
        >
          Facts
        </motion.h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          {STATS.map((stat, i) => <StatCard key={stat.label} stat={stat} index={i} />)}
        </div>
      </div>
    </section>
  )
}
