import { motion } from 'framer-motion'
import Footer from '../components/Footer'

function Reveal({ children, delay = 0, y = 28, style = {} }) {
  return (
    <motion.div
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ type: 'spring', stiffness: 55, damping: 20, delay }}
      style={style}
    >
      {children}
    </motion.div>
  )
}

const IMPACT = [
  { amount: '$25', desc: 'Covers sensor calibration supplies for one monitoring site for a month.' },
  { amount: '$100', desc: 'Helps purchase a low-cost air quality sensor deployed in a frontline neighborhood.' },
  { amount: '$500', desc: 'Funds a full data collection and outreach campaign in one community.' },
]

export default function DonatePage() {
  return (
    <div className="page-enter">
      {/* Header */}
      <section style={{
        background: 'linear-gradient(140deg, #1c3880 0%, #2952a8 30%, #3868c0 60%, #4c80d0 100%)',
        paddingTop: '130px',
        paddingBottom: '80px',
        paddingLeft: '80px',
        paddingRight: '80px',
      }}>
        <Reveal>
          <p style={{
            fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 600,
            color: 'rgba(255,255,255,0.6)', textTransform: 'uppercase',
            letterSpacing: '0.16em', marginBottom: '20px',
          }}>
            Support Our Work
          </p>
          <h1 style={{
            fontFamily: 'Lora, Georgia, serif',
            fontSize: 'clamp(32px, 4vw, 56px)', fontWeight: 600,
            color: 'white', lineHeight: 1.18,
            letterSpacing: '-0.025em', maxWidth: '680px',
            marginBottom: '24px',
          }}>
            Every dollar expands access to clean air.
          </h1>
          <p style={{
            fontFamily: 'Inter, sans-serif', fontSize: 'clamp(15px, 1.3vw, 17px)',
            color: 'rgba(255,255,255,0.72)', lineHeight: 1.72,
            maxWidth: '540px',
          }}>
            Your contribution helps us deploy sensors, train youth advocates, and bring real-time air quality data to the communities that need it most.
          </p>
        </Reveal>
      </section>

      {/* Impact tiers */}
      <section style={{ background: '#080f2a', padding: '80px 80px 60px' }}>
        <Reveal>
          <h2 style={{
            fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(20px, 2.2vw, 30px)',
            fontWeight: 600, color: 'white', letterSpacing: '-0.015em',
            marginBottom: '40px',
          }}>
            Your impact
          </h2>
        </Reveal>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', maxWidth: '900px' }}>
          {IMPACT.map((item, i) => (
            <Reveal key={item.amount} delay={i * 0.08}>
              <div style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '12px', padding: '32px 28px',
                backdropFilter: 'blur(8px)',
              }}>
                <p style={{
                  fontFamily: 'Lora, Georgia, serif', fontSize: '36px', fontWeight: 700,
                  color: 'white', marginBottom: '14px', lineHeight: 1,
                }}>
                  {item.amount}
                </p>
                <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '13.5px', color: 'rgba(255,255,255,0.6)', lineHeight: 1.72 }}>
                  {item.desc}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* GiveButter embed */}
      <section style={{ background: '#080f2a', padding: '0 80px 100px' }}>
        <Reveal>
          <div style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '16px',
            overflow: 'hidden',
            maxWidth: '900px',
          }}>
            <iframe
              src="https://givebutter.com/sharedskies"
              title="Donate to Shared Skies Initiative"
              width="100%"
              height="640"
              frameBorder="0"
              style={{ display: 'block', border: 'none' }}
            />
          </div>
        </Reveal>
      </section>

      <Footer />
    </div>
  )
}
