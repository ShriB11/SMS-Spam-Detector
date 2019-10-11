import React from 'react'
import { Button } from 'antd'

class App extends React.Component {
  render () {
    return (
      <div>
        <header>
          <div class='container'>
            <h2>SMS Spam Detector</h2>
          </div>
        </header>
        <div class='ml-container' />
        <Button type='primary'>Primary</Button>
      </div>
    )
  }
}
