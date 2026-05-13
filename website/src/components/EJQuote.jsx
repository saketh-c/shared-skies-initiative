import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

export default function EJQuote() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.3 })

  return (
    <section style={{
      background: 'linear-gradient(135deg, #2855c8 0%, #3b6fe0 40%, #4a80ef 70%, #3668d8 100%)',
      padding: '130px 84px 150px',
      minHeight: '68vh',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Subtle background texture */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 80% 60% at 20% 50%, rgba(255,255,255,0.07) 0%, transparent 100%)',
        pointerEvents: 'none',
      }} />

      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 44 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        style={{ position: 'relative', zIndex: 1 }}
      >
        <p style={{
          fontFamily: 'Lora, Georgia, serif',
          fontSize: 'clamp(22px, 3vw, 42px)',
          fontWeight: 500,
          color: 'white',
          maxWidth: '780px',
          lineHeight: 1.38,
          marginBottom: '52px',
          letterSpacing: '-0.015em',
        }}>
          Environmental justice is the promise that no one's health is determined by their ZIP code.
        </p>

        <motion.a
          href="https://form.typeform.com/to/FeqteoZD?typeform-source=sharedskiesinitiative.org"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ scale: 1.04, background: '#f0f5ff' }}
          whileTap={{ scale: 0.97 }}
          transition={{ duration: 0.18 }}
          style={{
            display: 'inline-block',
            background: 'white',
            color: '#1a3080',
            fontSize: '14px',
            fontWeight: 600,
            padding: '14px 30px',
            borderRadius: '6px',
            fontFamily: 'Inter, sans-serif',
            letterSpacing: '0.01em',
          }}
        >
          Let's work together
        </motion.a>
      </motion.div>
    </section>
  )
}
