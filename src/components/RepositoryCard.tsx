import React from 'react'
import { Repository } from 'react-github-embed'

const RepositoryCard = () => {
  return (
    <div>
      <Repository
        username="facebook"
        repository="react"
        theme="light"
        showStarCount={true}
        showForkCount={true}
        showLanguage={true}
        showDescription={true}
        showType={true}
      />
      <Repository username="Mridul2820" repository="css-js" theme="light" />
      <Repository
        username="Mridul2820"
        repository="next-template"
        theme="dark"
      />
    </div>
  )
}

export default RepositoryCard
