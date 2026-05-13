import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, useInView } from 'framer-motion'

export default function Collaborate() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.3 })

  return (
    <section style={{ background: '#1e3d6e', padding: '60px 68px 110px' }}>
      <motion.div
        ref={ref}
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 1 } : {}}
        transition={{ type: 'spring', stiffness: 55, damping: 20 }}
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '88px', alignItems: 'start' }}
      >
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ type: 'spring', stiffness: 55, damping: 20 }}
        >
          <h2 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(22px, 2.4vw, 34px)', fontWeight: 500, color: 'white', marginBottom: '32px', lineHeight: 1.3, letterSpacing: '-0.015em' }}>
            Want to build with us?
          </h2>
          <motion.div
            whileHover={{ background: 'rgba(255,255,255,0.14)' }}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.18 }}
            style={{ display: 'inline-block' }}
          >
            <Link
              to="/join"
              style={{
                display: 'inline-block',
                border: '1.5px solid rgba(255,255,255,0.55)',
                color: 'white', padding: '13px 30px',
                borderRadius: '6px', fontSize: '14px',
                fontFamily: 'Inter, sans-serif', fontWeight: 500,
                background: 'transparent', letterSpacing: '0.01em',
              }}
            >
              Let's start a project
            </Link>
          </motion.div>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 28 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ type: 'spring', stiffness: 55, damping: 20, delay: 0.1 }}
          style={{ fontFamily: 'Inter, sans-serif', fontSize: '14px', color: 'rgba(255,255,255,0.7)', lineHeight: 1.85 }}
        >
          We're always looking to collaborate with schools, researchers, community organizations, and advocates who share our commitment to clean air and environmental equity. Whether you're interested in hosting air-quality sensors, co-developing educational programs, or partnering on data and outreach projects, we'd love to connect. Together, we can expand access to real-time air data, amplify community voices, and build a future where every neighborhood can breathe easier under our shared skies.
        </motion.p>
      </motion.div>
    </section>
  )
}
