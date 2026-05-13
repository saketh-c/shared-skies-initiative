import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

const ITEMS = [
  {
    num: '01',
    title: 'Research',
    desc: 'We install and maintain accurate, low-cost air quality sensors in frontline neighborhoods. We create a real-time picture of local pollution and make information accessible to every resident.',
    img: 'https://images.unsplash.com/photo-1569163139394-de4e5f43e5ca?auto=format&fit=crop&w=700&q=80',
    imgBg: '#0c1868',
  },
  {
    num: '02',
    title: 'Education',
    desc: 'We influence daily decisions and behavior through education. Students, young and old, at any level, can understand social and environmental justice topics. We distribute curriculum and host workshops to foster a general understanding of this critical issue.',
    img: 'https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?auto=format&fit=crop&w=700&q=80',
    imgBg: '#0c1e5a',
  },
  {
    num: '03',
    title: 'Advocacy',
    desc: 'We advocate for local policy change using data-driven insights from our monitors. Youth ambassadors and community members receive knowledge and skills to interpret air data, share stories, and drive meaningful change.',
    img: 'https://images.unsplash.com/photo-1590650153855-d9e808231d41?auto=format&fit=crop&w=700&q=80',
    imgBg: '#081050',
  },
]

function ProcessRow({ item }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.2 })

  return (
    <div style={{
      display: 'flex',
      borderTop: '1px solid rgba(255,255,255,0.09)',
      minHeight: '470px',
    }}>
      <motion.div
        initial={{ opacity: 0, scale: 1.05 }}
        animate={inView ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
        style={{ width: '46%', flexShrink: 0, background: item.imgBg, overflow: 'hidden' }}
      >
        <img src={item.img} alt={item.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </motion.div>

      <motion.div
        ref={ref}
        initial={{ opacity: 0, x: 28 }}
        animate={inView ? { opacity: 1, x: 0 } : {}}
        transition={{ duration: 0.95, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}
        style={{ flex: 1, padding: '56px 64px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '22px', marginBottom: '24px' }}>
          <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '12px', color: 'rgba(255,255,255,0.38)', letterSpacing: '0.08em' }}>
            {item.num}
          </span>
          <h3 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '30px', fontWeight: 500, color: 'white', letterSpacing: '-0.01em' }}>
            {item.title}
          </h3>
        </div>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '14px', color: 'rgba(255,255,255,0.65)', lineHeight: 1.78, maxWidth: '380px' }}>
          {item.desc}
        </p>
      </motion.div>
    </div>
  )
}

export default function Process() {
  const headerRef = useRef(null)
  const headerInView = useInView(headerRef, { once: true, amount: 0.4 })

  return (
    <section style={{ background: '#0b1458' }}>
      <motion.div
        ref={headerRef}
        initial={{ opacity: 0, y: 24 }}
        animate={headerInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          padding: '64px 68px 52px',
          borderTop: '1px solid rgba(255,255,255,0.12)',
          gap: '48px',
        }}
      >
        <h2 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '30px', fontWeight: 500, color: 'white', flexShrink: 0, letterSpacing: '-0.01em' }}>
          Process
        </h2>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '14px', color: 'rgba(255,255,255,0.6)', maxWidth: '420px', lineHeight: 1.72, textAlign: 'right' }}>
          Our work is guided by three core values that ensure the highest quality and impact in every project we undertake.
        </p>
      </motion.div>

      {ITEMS.map(item => <ProcessRow key={item.num} item={item} />)}
    </section>
  )
}
