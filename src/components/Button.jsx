import React from 'react';
import './Button.css';

const Button = ({ 
    children, 
    onClick, 
    variant = 'primary', 
    size = 'md', 
    icon, 
    loading = false, 
    disabled = false, 
    className = '',
    type = 'button',
    ...props 
}) => {
    const buttonClasses = [
        'btn-component',
        `btn-${variant}`,
        `btn-size-${size}`,
        loading ? 'btn-loading' : '',
        className
    ].filter(Boolean).join(' ');

    return (
        <button
            type={type}
            className={buttonClasses}
            onClick={onClick}
            disabled={disabled || loading}
            {...props}
        >
            {loading && <span className="btn-spinner"></span>}
            {!loading && icon && <span className="btn-icon">{icon}</span>}
            <span className="btn-text">{children}</span>
        </button>
    );
};

export default Button;
