export default function MapPage() {
  return (
    <div style={{ paddingTop: '72px', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <iframe
        src="https://shared-skies-initiative.vercel.app/"
        title="Shared Skies Real-Time Air Quality Map"
        width="100%"
        style={{ flex: 1, border: 'none', display: 'block' }}
        allow="geolocation"
      />
    </div>
  )
}
