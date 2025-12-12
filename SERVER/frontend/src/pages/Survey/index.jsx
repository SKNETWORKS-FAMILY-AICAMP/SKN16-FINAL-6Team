import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../services/api'
import surveyData from '../../data/surveyQuestions.json'
import './Survey.css'

function Survey() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [currentExperience, setCurrentExperience] = useState('intermediate')
  const [formData, setFormData] = useState({
    programming_experience: 'intermediate',
    learning_goals: [],
    interested_topics: [],
    weak_topics: [],
    target_type: 'daily',
    daily_problem_goal: 2,
    weekly_problem_goal: 14
  })

  const { questions } = surveyData

  // 경험 레벨 변경 시
  const handleExperienceChange = (value) => {
    setCurrentExperience(value)
    setFormData(prev => ({
      ...prev,
      programming_experience: value
    }))
  }

  // 다중 선택 토글
  const handleMultipleToggle = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter(v => v !== value)
        : [...prev[field], value]
    }))
  }

  // 단일 선택
  const handleSingleSelect = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  // 슬라이더 변경
  const handleSliderChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: parseInt(value)
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (formData.learning_goals.length === 0) {
      alert('학습 목표를 최소 1개 이상 선택해주세요.')
      return
    }

    if (formData.interested_topics.length === 0) {
      alert('관심 분야를 최소 1개 이상 선택해주세요.')
      return
    }

    setLoading(true)
    try {
      const response = await api.post('/coding-test/survey/', formData)

      if (response.data.success) {
        alert('설문조사가 완료되었습니다! 맞춤 로드맵이 생성되었습니다.')
        navigate('/app/roadmap')
      } else {
        alert('설문조사 제출에 실패했습니다: ' + response.data.message)
      }
    } catch (error) {
      console.error('Survey submission error:', error)
      alert('설문조사 제출 중 오류가 발생했습니다.')
    } finally {
      setLoading(false)
    }
  }

  // 질문 1: 프로그래밍 경험
  const renderExperienceQuestion = (question) => (
    <div className="survey-section" key={question.id}>
      <h3>{question.question}</h3>
      <p className="section-description">{question.description}</p>
      <div className="radio-group">
        {question.options.map(option => (
          <label
            key={option.value}
            className={`radio-option ${formData.programming_experience === option.value ? 'selected' : ''}`}
          >
            <input
              type="radio"
              name="programming_experience"
              value={option.value}
              checked={formData.programming_experience === option.value}
              onChange={() => handleExperienceChange(option.value)}
            />
            <div className="option-content">
              <div className="option-title">{option.label}</div>
              <div className="option-description">{option.description}</div>
            </div>
          </label>
        ))}
      </div>
    </div>
  )

  // 질문 2: 학습 목표 (다중 선택)
  const renderGoalsQuestion = (question) => (
    <div className="survey-section" key={question.id}>
      <h3>{question.question}</h3>
      <p className="section-description">{question.description}</p>
      <div className="checkbox-grid">
        {question.options.map(option => (
          <label
            key={option.value}
            className={`checkbox-option ${formData.learning_goals.includes(option.value) ? 'selected' : ''}`}
          >
            <input
              type="checkbox"
              checked={formData.learning_goals.includes(option.value)}
              onChange={() => handleMultipleToggle('learning_goals', option.value)}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    </div>
  )

  // 질문 3, 4: 관심 분야 / 보완 분야 (다중 선택)
  const renderTopicsQuestion = (question, field) => {
    const options = question.options || []
    const selectedValues = formData[field]

    return (
      <div className="survey-section" key={question.id}>
        <h3>{question.question}</h3>
        <p className="section-description">{question.description}</p>

        <div className="checkbox-grid topics-grid">
          {options.map(option => (
            <label
              key={option.value}
              className={`checkbox-option topic-option ${selectedValues.includes(option.value) ? 'selected' : ''} ${option.difficulty ? `difficulty-${option.difficulty}` : ''}`}
            >
              <input
                type="checkbox"
                checked={selectedValues.includes(option.value)}
                onChange={() => handleMultipleToggle(field, option.value)}
              />
              <div className="topic-content">
                <span className="topic-icon">{option.icon}</span>
                <span className="topic-label">{option.label}</span>
                {option.difficulty && <span className="topic-difficulty">{option.difficulty}</span>}
              </div>
            </label>
          ))}
        </div>
      </div>
    )
  }

  // 질문 5: 학습 목표 설정 (슬라이더)
  const renderTargetQuestion = (question) => {
    const currentSlider = question.sliders[formData.target_type]

    return (
      <div className="survey-section" key={question.id}>
        <h3>{question.question}</h3>
        <p className="section-description">{question.description}</p>

        {/* 하루/주간 선택 */}
        <div className="target-type-selector">
          {question.select_options.map(option => (
            <button
              key={option.value}
              type="button"
              className={`target-type-btn ${formData.target_type === option.value ? 'active' : ''}`}
              onClick={() => handleSingleSelect('target_type', option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>

        {/* 슬라이더 */}
        <div className="slider-container">
          <div className="slider-header">
            <span>{currentSlider.label}</span>
            <span className="slider-value">
              {formData.target_type === 'daily' ? formData.daily_problem_goal : formData.weekly_problem_goal}
              {currentSlider.unit}
            </span>
          </div>
          <input
            type="range"
            min={currentSlider.min}
            max={currentSlider.max}
            step={currentSlider.step}
            value={formData.target_type === 'daily' ? formData.daily_problem_goal : formData.weekly_problem_goal}
            onChange={(e) => handleSliderChange(
              formData.target_type === 'daily' ? 'daily_problem_goal' : 'weekly_problem_goal',
              e.target.value
            )}
            className="goal-slider"
          />
          <div className="slider-labels">
            <span>{currentSlider.min}{currentSlider.unit}</span>
            <span>{currentSlider.max}{currentSlider.unit}</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="survey-page">
      <div className="survey-header">
        <h1>{surveyData.survey_title}</h1>
        <p>{surveyData.survey_description}</p>
      </div>

      <form className="survey-form" onSubmit={handleSubmit}>
        {questions.map(question => {
          switch (question.id) {
            case 1:
              return renderExperienceQuestion(question)
            case 2:
              return renderGoalsQuestion(question)
            case 3:
              return renderTopicsQuestion(question, 'interested_topics')
            case 4:
              return renderTopicsQuestion(question, 'weak_topics')
            case 5:
              return renderTargetQuestion(question)
            default:
              return null
          }
        })}

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? '제출 중...' : '설문 완료 및 로드맵 생성'}
        </button>
      </form>
    </div>
  )
}

export default Survey
