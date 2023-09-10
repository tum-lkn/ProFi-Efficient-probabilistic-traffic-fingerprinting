-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: mariadb
-- Erstellungszeit: 16. Mai 2020 um 13:09
-- Server-Version: 10.4.12-MariaDB-1:10.4.12+maria~bionic
-- PHP-Version: 7.4.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Datenbank: `swc_tls_gatherer`
--

-- --------------------------------------------------------

--
-- Tabellenstruktur für Tabelle `companies`
--

CREATE TABLE `companies` (
  `id` int(11) NOT NULL,
  `name` varchar(20) COLLATE utf32_german2_ci NOT NULL,
  `alexa_rank` int(5) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_german2_ci;

-- --------------------------------------------------------

--
-- Tabellenstruktur für Tabelle `tags`
--

CREATE TABLE `tags` (
  `id` int(11) NOT NULL,
  `tag` varchar(20) COLLATE utf32_german2_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_german2_ci;

-- --------------------------------------------------------

--
-- Tabellenstruktur für Tabelle `tag_assignment`
--

CREATE TABLE `tag_assignment` (
  `id` int(11) NOT NULL,
  `tag_fk` int(11) NOT NULL,
  `trace_fk` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_german2_ci;

-- --------------------------------------------------------

--
-- Tabellenstruktur für Tabelle `traces_metadata`
--

CREATE TABLE `traces_metadata` (
  `id` int(11) NOT NULL,
  `url` int(11) NOT NULL,
  `ipv4` varchar(15) COLLATE utf32_german2_ci NOT NULL,
  `filename` varchar(40) COLLATE utf32_german2_ci NOT NULL,
  `start_capture` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `end_capture` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00',
  `browser` varchar(40) COLLATE utf32_german2_ci NOT NULL,
  `os` varchar(20) COLLATE utf32_german2_ci NOT NULL,
  `tls_version` varchar(8) COLLATE utf32_german2_ci NOT NULL,
  `was_successful` tinyint(1) DEFAULT NULL,
  `is_resumption` tinyint(1) NOT NULL,
  `added_by` varchar(15) COLLATE utf32_german2_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_german2_ci;

-- --------------------------------------------------------

--
-- Tabellenstruktur für Tabelle `urls`
--

CREATE TABLE `urls` (
  `id` int(11) NOT NULL,
  `url` varchar(650) COLLATE utf32_german2_ci NOT NULL,
  `company` int(11) NOT NULL,
  `tags` varchar(100) COLLATE utf32_german2_ci NOT NULL,
  `trace_done` tinyint(1) NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_german2_ci;

--
-- Indizes der exportierten Tabellen
--

--
-- Indizes für die Tabelle `companies`
--
ALTER TABLE `companies`
  ADD PRIMARY KEY (`id`);

--
-- Indizes für die Tabelle `tags`
--
ALTER TABLE `tags`
  ADD PRIMARY KEY (`id`);

--
-- Indizes für die Tabelle `tag_assignment`
--
ALTER TABLE `tag_assignment`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_tag` (`tag_fk`) USING BTREE,
  ADD KEY `fk_traces` (`trace_fk`);

--
-- Indizes für die Tabelle `traces_metadata`
--
ALTER TABLE `traces_metadata`
  ADD PRIMARY KEY (`id`),
  ADD KEY `url_fk` (`url`) USING BTREE;

--
-- Indizes für die Tabelle `urls`
--
ALTER TABLE `urls`
  ADD PRIMARY KEY (`id`),
  ADD KEY `company` (`company`) USING BTREE;

--
-- AUTO_INCREMENT für exportierte Tabellen
--

--
-- AUTO_INCREMENT für Tabelle `companies`
--
ALTER TABLE `companies`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT für Tabelle `tags`
--
ALTER TABLE `tags`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT für Tabelle `tag_assignment`
--
ALTER TABLE `tag_assignment`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT für Tabelle `traces_metadata`
--
ALTER TABLE `traces_metadata`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT für Tabelle `urls`
--
ALTER TABLE `urls`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Constraints der exportierten Tabellen
--

--
-- Constraints der Tabelle `tag_assignment`
--
ALTER TABLE `tag_assignment`
  ADD CONSTRAINT `tag_assignment_ibfk_1` FOREIGN KEY (`tag_fk`) REFERENCES `tags` (`id`) ON UPDATE NO ACTION,
  ADD CONSTRAINT `tag_assignment_ibfk_2` FOREIGN KEY (`trace_fk`) REFERENCES `traces_metadata` (`id`) ON UPDATE NO ACTION;

--
-- Constraints der Tabelle `traces_metadata`
--
ALTER TABLE `traces_metadata`
  ADD CONSTRAINT `traces_metadata_ibfk_1` FOREIGN KEY (`url`) REFERENCES `urls` (`id`) ON UPDATE CASCADE;

--
-- Constraints der Tabelle `urls`
--
ALTER TABLE `urls`
  ADD CONSTRAINT `company` FOREIGN KEY (`company`) REFERENCES `companies` (`id`) ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
