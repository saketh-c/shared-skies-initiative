import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

export default function AboutIntro() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.3 })

  return (
    <section id="about" style={{
      background: `linear-gradient(180deg,
        #f5faff 0%,
        #e8f2fc 20%,
        #d4e6f8 45%,
        #c2d8f5 68%,
        #edf5ff 88%,
        #ffffff 100%
      )`,
      padding: '130px 80px 150px',
      minHeight: '75vh',
      display: 'flex',
      alignItems: 'center',
    }}>
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 50 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 1.05, ease: [0.22, 1, 0.36, 1] }}
        style={{ maxWidth: '900px' }}
      >
        <p style={{
          fontFamily: 'Lora, Georgia, serif',
          fontSize: 'clamp(20px, 2.4vw, 32px)',
          fontWeight: 500,
          color: '#122470',
          lineHeight: 1.55,
          marginBottom: '40px',
          letterSpacing: '-0.01em',
        }}>
          We work with frontline neighborhoods and provide the data, resources, and advocacy they need to fight air pollution and secure a healthier future.
        </p>
        <motion.span
          initial={{ scale: 0, opacity: 0 }}
          animate={inView ? { scale: 1, opacity: 1 } : {}}
          transition={{ delay: 0.7, duration: 0.45, ease: 'backOut' }}
          style={{
            display: 'inline-block',
            width: '9px', height: '9px',
            borderRadius: '50%',
            background: '#122470',
          }}
        />
      </motion.div>
    </section>
  )
}
